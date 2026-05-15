const std = @import("std");
const StrideWalker = @import("stridewalker.zig");

threadlocal var context: ?*Context = null;

pub const Context = struct {
    gpa: std.mem.Allocator,
    dataPack: std.ArrayList(f32),
    dataGradPack: std.ArrayList(f32),
    parameterPack: std.ArrayList(f32),
    parameterGradPack: std.ArrayList(f32),
    parameterCursor: u32,
    history: std.ArrayList(u8),
    initializer: Initializer,
    rand: std.Random.DefaultPrng,
    usingCheck: std.atomic.Value(bool),

    pub fn init(gpa: std.mem.Allocator, seed: u64) Context {
        return .{
            .rand = .init(seed), // random seed
            .gpa = gpa,
            .dataPack = .empty,
            .dataGradPack = .empty,
            .parameterPack = .empty,
            .parameterGradPack = .empty,
            .parameterCursor = 0,
            .history = .empty,
            .initializer = .uniform,
            .usingCheck = .init(false),
        };
    }
    pub fn deinit(ctx: *Context) void {
        if (context == ctx) {
            context = null;
        }
        ctx.parameterPack.deinit(ctx.gpa);
        ctx.parameterGradPack.deinit(ctx.gpa);
        ctx.dataPack.deinit(ctx.gpa);
        ctx.dataGradPack.deinit(ctx.gpa);
        ctx.history.deinit(ctx.gpa);
        ctx.* = undefined;
    }

    // return false if already using
    pub fn set(ctx: *Context) bool {
        if (ctx.usingCheck.cmpxchgStrong(false, true, .monotonic, .monotonic) != null) {
            return false;
        }
        if (context) |old| {
            old.usingCheck.store(false, .unordered);
        }
        context = ctx;
        return true;
    }
    pub fn reset() void {
        if (context) |old| {
            old.usingCheck.store(false, .unordered);
            context = null;
        }
    }

    fn gradDataOffset(ctx: *Context) GradDataOffset {
        return .{ .bytes = @intFromPtr(ctx.dataGradPack.items.ptr) -% @intFromPtr(ctx.dataPack.items.ptr) };
    }
    fn createTensor(ctx: *Context, values: []const f32, dataLen: u32, batchLen: u32) !Tensor {
        std.debug.assert(dataLen * batchLen == values.len);
        const dataIdx = ctx.dataPack.items.len;
        try ctx.dataPack.appendSlice(ctx.gpa, values);
        return .{ .dataIdx = @intCast(dataIdx), .dataLen = dataLen, .batchStride = dataLen, .batchLen = batchLen };
    }
    fn createDirtyTensor(ctx: *Context, dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        const totalLen = dataLen * batchLen;
        const dataIdx = ctx.dataPack.items.len;
        const array = try ctx.dataPack.addManyAsSlice(ctx.gpa, totalLen);
        return .{
            .{ .dataIdx = @intCast(dataIdx), .dataLen = dataLen, .batchStride = dataLen, .batchLen = batchLen },
            array,
        };
    }

    fn generateInputTensor(ctx: *Context, data: []const DataPair) !Tensor {
        std.debug.assert(data.len != 0);
        const inputLen: u32 = @intCast(data[0].input.len);
        const t, const dest = try ctx.createDirtyTensor(inputLen, @intCast(data.len));
        var destPtr = dest.ptr;
        for (data) |d| {
            @memcpy(destPtr[0..inputLen], d.input);
            destPtr += inputLen;
        }
        return t;
    }
    fn generateLabelTensor(ctx: *Context, data: []const DataPair) !Tensor {
        std.debug.assert(data.len != 0);
        const outputLen: u32 = @intCast(data[0].label.len);
        const t, const dest = try ctx.createDirtyTensor(outputLen, @intCast(data.len));
        var destPtr = dest.ptr;
        for (data) |d| {
            @memcpy(destPtr[0..outputLen], d.label);
            destPtr += outputLen;
        }
        return t;
    }
};

pub const GradDataOffset = struct {
    bytes: usize,

    pub fn ptr(gdo: GradDataOffset, p: [*]f32) [*]f32 {
        return @ptrFromInt(@intFromPtr(p) +% gdo.bytes);
    }
    pub fn slice(gdo: GradDataOffset, s: []f32) []f32 {
        const out: [*]f32 = @ptrFromInt(@intFromPtr(s.ptr) +% gdo.bytes);
        return out[0..s.len];
    }
};

pub const TensorData = struct {
    data: [*]f32,
    dataLen: u32,
    batchStride: u32,
    batchLen: u32,

    pub fn format(
        t: TensorData,
        writer: *std.Io.Writer,
    ) !void {
        try writer.writeByte('{');
        if (t.batchLen > 0) {
            var ptr = t.data;
            try writer.print("{any}", .{ptr[0..t.dataLen]});
            const dataEnd = ptr + t.batchLen * t.batchStride;
            ptr += t.batchStride;
            while (@intFromPtr(ptr) < @intFromPtr(dataEnd)) {
                try writer.writeByte(',');
                try writer.print("{any}", .{ptr[0..t.dataLen]});
                ptr += t.batchStride;
            }
        }
        try writer.writeByte('}');
    }

    pub fn copyTo(t: TensorData, dest: [*]f32) void {
        var srcPtr = t.data;
        const srcEnd = srcPtr + t.batchLen * t.batchStride;
        var destPtr = dest;
        while (@intFromPtr(srcPtr) < @intFromPtr(srcEnd)) {
            @memcpy(destPtr[0..t.dataLen], srcPtr[0..t.dataLen]);
            srcPtr += t.batchStride;
            destPtr += t.dataLen;
        }
    }
    pub fn copyFrom(t: TensorData, src: [*]const f32) void {
        var destPtr = t.data;
        const destEnd = destPtr + t.batchLen * t.batchStride;
        var srcPtr = src;
        while (@intFromPtr(destPtr) < @intFromPtr(destEnd)) {
            @memcpy(destPtr[0..t.dataLen], srcPtr[0..t.dataLen]);
            srcPtr += t.dataLen;
            destPtr += t.batchStride;
        }
    }
    fn testingApproxEq(t: TensorData, expected: []const f32) !void {
        var walker = StrideWalker.init(t.data, t.dataLen, t.batchStride, t.batchLen);
        return walker.testingApproxEq(expected);
    }
};

const Initializer = enum {
    uninit,
    uniform,
};

const BackwardFn = *const fn () void;

pub const TensorScope = struct {
    inputBegin: usize,

    pub fn save() TensorScope {
        const ctx = context orelse unreachable;
        return .{ .inputBegin = ctx.dataPack.items.len };
    }
    pub fn restore(s: TensorScope) void {
        const ctx = context orelse unreachable;
        std.debug.assert(ctx.dataPack.items.len >= s.inputBegin);
        ctx.dataPack.items.len = s.inputBegin;
    }
};

pub const DataPair = struct {
    input: []const f32,
    label: []const f32,
};

pub fn createNetwork(T: type) !Network(T) {
    return Network(T).init();
}

pub fn Network(T: type) type {
    return struct {
        parameterBegin: u32,
        parameterLen: u32,
        implement: *const fn (xs: Tensor) anyerror!Tensor,
        data: T,

        pub fn init() !@This() {
            const ctx = context orelse unreachable;

            const oldInit = ctx.initializer;
            defer ctx.initializer = oldInit;
            ctx.initializer = .uninit;

            const scope = TensorScope.save();
            defer scope.restore();

            const xs, _ = try ctx.createDirtyTensor(T.inputLen, 1);
            var n: @This() = .{
                .parameterBegin = undefined,
                .parameterLen = undefined,
                .implement = T.forward,
                .data = .{},
            };
            n.parameterBegin = @intCast(ctx.parameterPack.items.len);
            ctx.parameterCursor = n.parameterBegin;
            ctx.history.clearRetainingCapacity();
            _ = try n.implement(xs);
            n.parameterLen = @intCast(ctx.parameterPack.items.len - n.parameterBegin);
            return n;
        }
        pub fn predict(n: *@This(), input: []const f32) ![]f32 {
            const ctx = context orelse unreachable;
            if (input.len == 0) {
                return &.{};
            }

            const scope = TensorScope.save();
            defer scope.restore();

            const batchLen: u32 = @intCast(input.len / T.inputLen);
            const xs = try ctx.createTensor(input, T.inputLen, batchLen);
            const ys = try n.predictWithTensor(xs);
            return ys.plainData();
        }

        pub fn predictWithTensor(n: *@This(), xs: Tensor) !Tensor {
            const ctx = context orelse unreachable;
            if (xs.dataLen != T.inputLen) return Error.SizeMismatch;

            ctx.parameterCursor = n.parameterBegin;
            ctx.history.clearRetainingCapacity();

            const ys = try n.implement(xs);
            const parameterLen: u32 = @intCast(ctx.parameterPack.items.len - n.parameterBegin);
            std.debug.assert(parameterLen == n.parameterLen);
            return ys;
        }

        // return loss
        pub fn trainOnceWithTensor(n: *@This(), xs: Tensor, labels: Tensor, learningRate: f32) !Tensor {
            const ys = try n.predictWithTensor(xs);
            const loss = try ys.l2Loss(labels);
            try loss.backward(&.{1.0});
            sgd(learningRate);
            return loss;
        }

        pub fn train(n: *@This(), data: []const DataPair, learningRate: f32, epoch: u32) !void {
            const ctx = context orelse unreachable;
            if (data.len == 0) return;

            const scope = TensorScope.save();
            defer scope.restore();

            const xs = try ctx.generateInputTensor(data);
            const labels = try ctx.generateLabelTensor(data);

            for (0..epoch) |_| {
                const scope2 = TensorScope.save();
                defer scope2.restore();

                _ = try n.trainOnceWithTensor(xs, labels, learningRate);
            }
        }

        pub fn parameters(n: *@This()) []f32 {
            const ctx = context orelse unreachable;
            return ctx.parameterPack.items[n.parameterBegin .. n.parameterBegin + n.parameterLen];
        }
        pub fn parameterGradients(n: *@This()) []f32 {
            const ctx = context orelse unreachable;
            std.debug.assert(ctx.parameterGradPack.items.len >= n.parameterBegin + n.parameterLen); // backward required
            return ctx.parameterGradPack.items[n.parameterBegin .. n.parameterBegin + n.parameterLen];
        }
    };
}

pub const Error = error{SizeMismatch};

pub const Tensor = struct {
    dataIdx: u32,
    dataLen: u32,
    batchStride: u32,
    batchLen: u32,

    pub fn init(values: []const f32, dataLen: u32, batchLen: u32) !Tensor {
        const ctx = context orelse unreachable;
        return ctx.createTensor(values, dataLen, batchLen);
    }
    pub fn initUndef(dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        const ctx = context orelse unreachable;
        return ctx.createDirtyTensor(dataLen, batchLen);
    }
    pub fn dataPtr(t: Tensor) [*]f32 {
        const ctx = context orelse unreachable;
        return ctx.dataPack.items.ptr + t.dataIdx;
    }
    fn assertGradient(t: Tensor) void {
        const ctx = context orelse unreachable;
        std.debug.assert(ctx.dataGradPack.items.len >= t.dataIdx + t.dataLen); // backward required
    }
    pub fn gradientPtr(t: Tensor) [*]f32 {
        t.assertGradient();
        const ctx = context orelse unreachable;
        return ctx.dataGradPack.items.ptr + t.dataIdx;
    }
    pub fn dataWalker(t: Tensor) StrideWalker {
        return .init(t.dataPtr(), t.dataLen, t.batchStride, t.batchLen);
    }
    pub fn gradientWalker(t: Tensor) StrideWalker {
        t.assertGradient();
        return .init(t.gradientPtr(), t.dataLen, t.batchStride, t.batchLen);
    }
    pub fn dataWalkerRow(t: Tensor) StrideWalker.Row {
        return .init(t.dataPtr(), t.dataLen, t.batchStride, t.batchLen);
    }
    pub fn gradientWalkerRow(t: Tensor) StrideWalker.Row {
        t.assertGradient();
        return .init(t.gradientPtr(), t.dataLen, t.batchStride, t.batchLen);
    }
    pub fn plainData(t: Tensor) ![]f32 {
        if (t.batchLen == 1) {
            return t.dataPtr()[0..t.dataLen];
        }
        if (t.batchStride == t.dataLen) {
            return t.dataPtr()[0 .. t.batchStride * t.batchLen];
        }

        _, const dest = try Tensor.initUndef(t.dataLen, t.batchLen);
        t.data().copyTo(dest.ptr);
        return dest;
    }
    pub fn data(t: Tensor) TensorData {
        return .{ .data = t.dataPtr(), .dataLen = t.dataLen, .batchStride = t.batchStride, .batchLen = t.batchLen };
    }
    pub fn gradient(t: Tensor) TensorData {
        t.assertGradient();
        return .{ .data = t.gradientPtr(), .dataLen = t.dataLen, .batchStride = t.batchStride, .batchLen = t.batchLen };
    }
    pub fn backward(t: Tensor, grad: []const f32) !void {
        const ctx = context orelse unreachable;
        std.debug.assert(t.dataLen * t.batchLen == grad.len); // size mismatch

        try ctx.dataGradPack.resize(ctx.gpa, ctx.dataPack.items.len);
        try ctx.parameterGradPack.resize(ctx.gpa, ctx.parameterPack.items.len);

        @memset(ctx.dataGradPack.items, 0.0);
        @memset(ctx.parameterGradPack.items, 0.0);

        var dest = t.gradient();
        dest.copyFrom(grad.ptr);

        while (ctx.history.items.len > 0) {
            const backFn = readHistory(BackwardFn);
            backFn();
        }
    }
    pub fn subarray(t: Tensor, begin: u32, end: u32) Tensor {
        std.debug.assert(end <= t.dataLen);
        std.debug.assert(begin <= end);
        return .{
            .dataIdx = t.dataIdx + begin,
            .dataLen = end - begin,
            .batchStride = t.batchStride,
            .batchLen = t.batchLen,
        };
    }
    fn testingApproxEq(t: Tensor, expected: []const f32) !void {
        return t.data().testingApproxEq(expected);
    }

    pub fn linear(xt: Tensor, yLen: u32) !Tensor {
        const ctx = context orelse unreachable;
        const xLen = xt.dataLen;
        const xStride = xt.batchStride;
        const weights = yLen * (xLen + 1);
        const inputLenF32: f32 = @floatFromInt(xLen);
        const outputLenF32: f32 = @floatFromInt(yLen);
        const paramIdx = ctx.parameterCursor;
        const ps: []f32 = try getParameters(weights, std.math.sqrt(6.0 / (inputLenF32 + outputLenF32)));

        const xs = xt.dataPtr();
        const yt, const ys = try Tensor.initUndef(yLen, xt.batchLen);

        const pPtrEnd = ps.ptr + ps.len;
        var yPtr = ys.ptr;
        const yPtrEnd = ys.ptr + ys.len;
        var xPtr = xs;

        while (@intFromPtr(yPtr) < @intFromPtr(yPtrEnd)) {
            const inputs = xPtr[0..xLen];
            xPtr += xStride;

            var pPtr = ps.ptr;
            while (@intFromPtr(pPtr) < @intFromPtr(pPtrEnd)) {
                var ow = pPtr[0];
                pPtr += 1;
                for (inputs) |ival| {
                    ow += ival * pPtr[0];
                    pPtr += 1;
                }
                yPtr[0] = ow;
                yPtr += 1;
            }
        }
        try writeHistory(LinearBackward{ .xt = xt, .yIdx = yt.dataIdx, .yLen = yt.dataLen, .paramIdx = paramIdx });
        return yt;
    }
    const LinearBackward = struct {
        xt: Tensor,
        yIdx: u32,
        yLen: u32,
        paramIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            const xs = b.xt.dataPtr();
            const gxs = b.xt.gradientPtr();
            const xLen = b.xt.dataLen;
            const paramCols = xLen + 1;
            const xStride = b.xt.batchStride;
            const xNext = xStride - xLen;
            const yLen = b.yLen;
            const batchLen = b.xt.batchLen;
            const gys = ctx.dataGradPack.items.ptr + b.yIdx;
            const gyEnd = gys + yLen * batchLen;

            var gyPtr = gys;

            // weight/bias grads
            {
                const gps = ctx.parameterGradPack.items.ptr + b.paramIdx;
                var gpPtr = gps;
                const xRowEnd = xs + xLen;
                const gyRowEnd = gys + yLen;
                while (@intFromPtr(gyPtr) < @intFromPtr(gyRowEnd)) {
                    {
                        var gyColPtr = gyPtr;
                        var w: f32 = 0;
                        while (@intFromPtr(gyColPtr) < @intFromPtr(gyEnd)) {
                            w += gyColPtr[0];
                            gyColPtr += yLen;
                        }
                        gpPtr[0] += w;
                        gpPtr += 1;
                    }
                    var xPtr = xs;
                    while (@intFromPtr(xPtr) < @intFromPtr(xRowEnd)) {
                        var gyColPtr = gyPtr;
                        var xPtrInner = xPtr;
                        var w: f32 = 0;
                        while (@intFromPtr(gyColPtr) < @intFromPtr(gyEnd)) {
                            w += xPtrInner[0] * gyColPtr[0];
                            xPtrInner += xStride;
                            gyColPtr += yLen;
                        }
                        gpPtr[0] += w;
                        gpPtr += 1;
                        xPtr += 1;
                    }
                    gyPtr += 1;
                }
            }

            // x grads
            const gxEnd = gxs + xStride * batchLen;
            const ps = ctx.parameterPack.items.ptr + b.paramIdx;
            var gxPtr = gxs;
            gyPtr = gys;
            while (@intFromPtr(gxPtr) < @intFromPtr(gxEnd)) {
                var pPtr = ps + 1;
                const gxRowEnd = gxPtr + xLen;
                while (@intFromPtr(gxPtr) < @intFromPtr(gxRowEnd)) {
                    var w: f32 = 0;
                    const gyRowEnd = gyPtr + yLen;
                    var pColPtr = pPtr;
                    var gyRowPtr = gyPtr;
                    while (@intFromPtr(gyRowPtr) < @intFromPtr(gyRowEnd)) {
                        w += gyRowPtr[0] * pColPtr[0];
                        pColPtr += paramCols;
                        gyRowPtr += 1;
                    }
                    gxPtr[0] += w;
                    gxPtr += 1;
                    pPtr += 1;
                }
                gxPtr += xNext;
                gyPtr += yLen;
            }
        }
    };
    pub fn l2Loss(xt: Tensor, labelT: Tensor) !Tensor {
        var out: f32 = 0;
        const xs = xt.dataPtr();
        const labels = labelT.dataPtr();

        const xEnd = xs + xt.batchStride * xt.batchLen;
        const xNext = xt.batchStride - xt.dataLen;
        const labelNext = labelT.batchStride - labelT.dataLen;

        var xPtr = xs;
        var labelPtr = labels;
        while (@intFromPtr(xPtr) < @intFromPtr(xEnd)) {
            const xRowEnd = xPtr + xt.dataLen;
            while (@intFromPtr(xPtr) < @intFromPtr(xRowEnd)) {
                const diff = xPtr[0] - labelPtr[0];
                out += diff * diff;
                xPtr += 1;
                labelPtr += 1;
            }
            xPtr += xNext;
            labelPtr += labelNext;
        }
        const total: f32 = @floatFromInt(xt.dataLen * xt.batchLen);
        out /= total;

        const outTensor = try Tensor.init(&.{out}, 1, 1);
        try writeHistory(L2LossBackward{ .xt = xt, .labelT = labelT, .yIdx = outTensor.dataIdx });
        return outTensor;
    }
    const L2LossBackward = struct {
        xt: Tensor,
        labelT: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            const gy2 = ctx.dataGradPack.items.ptr[b.yIdx] * 2;
            var xWalker = b.xt.dataWalker();
            var labelWalker = b.labelT.dataWalker();
            const gdo = ctx.gradDataOffset();

            while (xWalker.next()) {
                std.debug.assert(labelWalker.next());
                const grad = (xWalker.ptr[0] - labelWalker.ptr[0]) * gy2;
                gdo.ptr(xWalker.ptr)[0] += grad;
                gdo.ptr(labelWalker.ptr)[0] -= grad;
            }
        }
    };
    pub fn neg(xt: Tensor) !Tensor {
        var xWalker = xt.dataWalker();
        const yt, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);
        var yPtr = ys.ptr;
        while (xWalker.next()) {
            yPtr[0] = -xWalker.ptr[0];
            yPtr += 1;
        }
        try writeHistory(NegBackward{
            .xt = xt,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const NegBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gxWalker = b.xt.gradientWalker();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (gxWalker.next()) {
                const gy = gyPtr[0];
                gyPtr += 1;
                gxWalker.ptr[0] -= gy;
            }
        }
    };
    pub fn add(x1t: Tensor, x2t: Tensor) !Tensor {
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try Tensor.initUndef(x1t.dataLen, x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] + x2Walker.ptr[0];
            yPtr += 1;
        }
        try writeHistory(AddBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const AddBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gx1Walker = b.x1t.gradientWalker();
            var gx2Walker = b.x2t.gradientWalker();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (gx1Walker.next()) {
                _ = gx2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                gx1Walker.ptr[0] += gy;
                gx2Walker.ptr[0] += gy;
            }
        }
    };
    pub fn sub(x1t: Tensor, x2t: Tensor) !Tensor {
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try Tensor.initUndef(x1t.dataLen, x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] - x2Walker.ptr[0];
            yPtr += 1;
        }
        try writeHistory(SubBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const SubBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gx1Walker = b.x1t.gradientWalker();
            var gx2Walker = b.x2t.gradientWalker();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (gx1Walker.next()) {
                _ = gx2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                gx1Walker.ptr[0] += gy;
                gx2Walker.ptr[0] -= gy;
            }
        }
    };
    pub fn mul(x1t: Tensor, x2t: Tensor) !Tensor {
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try Tensor.initUndef(x1t.dataLen, x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] * x2Walker.ptr[0];
            yPtr += 1;
        }
        try writeHistory(MulBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const MulBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var x1Walker = b.x1t.dataWalker();
            var x2Walker = b.x2t.dataWalker();
            const gdo = ctx.gradDataOffset();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (x1Walker.next()) {
                _ = x2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                gdo.ptr(x1Walker.ptr)[0] += x2Walker.ptr[0] * gy;
                gdo.ptr(x2Walker.ptr)[0] += x1Walker.ptr[0] * gy;
            }
        }
    };
    pub fn div(x1t: Tensor, x2t: Tensor) !Tensor {
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try Tensor.initUndef(x1t.dataLen, x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] / x2Walker.ptr[0];
            yPtr += 1;
        }
        try writeHistory(DivBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const DivBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var x1Walker = b.x1t.dataWalker();
            var x2Walker = b.x2t.dataWalker();
            const gdo = ctx.gradDataOffset();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (x1Walker.next()) {
                _ = x2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                const x2 = x2Walker.ptr[0];
                gdo.ptr(x1Walker.ptr)[0] += gy / x2;
                gdo.ptr(x2Walker.ptr)[0] += gy * -x1Walker.ptr[0] / (x2 * x2);
            }
        }
    };
    pub fn dot(x1t: Tensor, x2t: Tensor) !Tensor {
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalkerRow();
        var x2Walker = x2t.dataWalkerRow();
        const yt, const ys = try Tensor.initUndef(1, x1t.batchLen);

        var yPtr = ys.ptr;
        while (x1Walker.next()) |x1row| {
            const x2row = x2Walker.next().?;
            var sumVal: f32 = 0;

            for (x1row, x2row) |x1, x2| {
                sumVal += x1 * x2;
            }
            yPtr[0] = sumVal;
            yPtr += 1;
        }

        try writeHistory(DotBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const DotBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var x1Walker = b.x1t.dataWalkerRow();
            var x2Walker = b.x2t.dataWalkerRow();
            const gdo = ctx.gradDataOffset();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (x1Walker.next()) |x1row| {
                const x2row = x2Walker.next().?;

                const gy = gyPtr[0];
                gyPtr += 1;

                for (x1row, x2row, gdo.slice(x1row), gdo.slice(x2row)) |x1, x2, *gx1, *gx2| {
                    gx1.* += gy * x2;
                    gx2.* += gy * x1;
                }
            }
        }
    };
    pub fn sum(xt: Tensor) !Tensor {
        var xWalker = xt.dataWalkerRow();
        const yt, const ys = try Tensor.initUndef(1, xt.batchLen);

        var yPtr = ys.ptr;
        while (xWalker.next()) |xrow| {
            var sumVal: f32 = 0;
            for (xrow) |x1| {
                sumVal += x1;
            }
            yPtr[0] = sumVal;
            yPtr += 1;
        }

        try writeHistory(SumBackward{
            .xt = xt,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const SumBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gxWalker = b.xt.gradientWalkerRow();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (gxWalker.next()) |gxrow| {
                const gy = gyPtr[0];
                gyPtr += 1;
                for (gxrow) |*gx| {
                    gx.* += gy;
                }
            }
        }
    };
    pub fn mean(xt: Tensor) !Tensor {
        var xWalker = xt.dataWalkerRow();
        const yt, const ys = try Tensor.initUndef(1, xt.batchLen);
        const xLenF32: f32 = @floatFromInt(xt.dataLen);

        var yPtr = ys.ptr;
        while (xWalker.next()) |xrow| {
            var sumVal: f32 = 0;
            for (xrow) |x1| {
                sumVal += x1;
            }
            yPtr[0] = sumVal / xLenF32;
            yPtr += 1;
        }

        try writeHistory(MeanBackward{
            .xt = xt,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const MeanBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            const xLenF32: f32 = @floatFromInt(b.xt.dataLen);
            var gxWalker = b.xt.gradientWalkerRow();
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (gxWalker.next()) |gxrow| {
                const gy = gyPtr[0] / xLenF32;
                gyPtr += 1;
                for (gxrow) |*gx| {
                    gx.* += gy;
                }
            }
        }
    };
    pub fn relu(xt: Tensor) !Tensor {
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        while (xWalker.next()) {
            if (xWalker.ptr[0] > 0) {
                yPtr[0] = xWalker.ptr[0];
            } else {
                yPtr[0] = 0;
            }
            yPtr += 1;
        }
        try writeHistory(ReluBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
        });
        return out;
    }
    const ReluBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            var xWalker = b.xt.dataWalker();
            const gbo = ctx.gradDataOffset();

            while (xWalker.next()) {
                if (xWalker.ptr[0] > 0) {
                    gbo.ptr(xWalker.ptr)[0] += gyPtr[0];
                }
                gyPtr += 1;
            }
        }
    };
    pub fn leakyRelu(xt: Tensor, alpha: f32) !Tensor {
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        while (xWalker.next()) {
            if (xWalker.ptr[0] > 0) {
                yPtr[0] = xWalker.ptr[0];
            } else {
                yPtr[0] = xWalker.ptr[0] * alpha;
            }
            yPtr += 1;
        }
        try writeHistory(LeakyReluBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
            .alpha = alpha,
        });
        return out;
    }
    const LeakyReluBackward = struct {
        xt: Tensor,
        yIdx: u32,
        alpha: f32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            var xWalker = b.xt.dataWalker();
            const gbo = ctx.gradDataOffset();

            while (xWalker.next()) {
                const gy = gyPtr[0];
                gbo.ptr(xWalker.ptr)[0] += if (xWalker.ptr[0] > 0) gy else gy * b.alpha;
                gyPtr += 1;
            }
        }
    };
    pub fn sigmoid(xt: Tensor) !Tensor {
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        while (xWalker.next()) {
            const x = xWalker.ptr[0];
            if (x >= 0) {
                const z = std.math.exp(-x);
                yPtr[0] = 1.0 / (1.0 + z);
            } else {
                const z = std.math.exp(x);
                yPtr[0] = z / (1.0 + z);
            }
            // yPtr[0] = 1 / (1 + std.math.exp(x));
            yPtr += 1;
        }
        try writeHistory(SigmoidBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
        });
        return out;
    }
    const SigmoidBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var yPtr = ctx.dataPack.items.ptr + b.yIdx;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            var gxWalker = b.xt.gradientWalker();

            while (gxWalker.next()) {
                const y = yPtr[0];
                gxWalker.ptr[0] = gyPtr[0] * y * (1 - y);
                yPtr += 1;
                gyPtr += 1;
            }
        }
    };
    pub fn tanh(xt: Tensor) !Tensor {
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        while (xWalker.next()) {
            yPtr[0] = std.math.tanh(xWalker.ptr[0]);
            yPtr += 1;
        }
        try writeHistory(TanhBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
        });
        return out;
    }
    const TanhBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var yPtr = ctx.dataPack.items.ptr + b.yIdx;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            var gxWalker = b.xt.gradientWalker();

            while (gxWalker.next()) {
                const y = yPtr[0];
                gxWalker.ptr[0] = gyPtr[0] * (1 - y * y);
                yPtr += 1;
                gyPtr += 1;
            }
        }
    };

    // TODO: exp, log, clamp, abs, sqrt

    // TODO: mse, mae, binary cross entropy

    // TODO: max, min
};

fn u32count(T: type) usize {
    return @divTrunc(@sizeOf(T) + @alignOf(u32) - 1, @alignOf(u32));
}
fn writeHistory(value: anytype) !void {
    const T = @TypeOf(value);

    const Impl = struct {
        fn call() void {
            const ctx = context orelse unreachable;
            const offset = ctx.history.items.len - @sizeOf(T);
            var valueSpace: T = undefined;
            @memcpy(std.mem.asBytes(&valueSpace), ctx.history.items[offset..]);
            ctx.history.items.len = offset;
            return valueSpace.backward();
        }
    };

    const ctx = context orelse unreachable;
    const buf = try ctx.history.addManyAsArray(ctx.gpa, @sizeOf(T) + @sizeOf(BackwardFn));
    @memcpy(buf[0..@sizeOf(T)], std.mem.asBytes(&value));
    const backf: BackwardFn = Impl.call;
    @memcpy(buf[@sizeOf(T)..], std.mem.asBytes(&backf));
}
fn readHistory(T: type) T {
    const ctx = context orelse unreachable;
    const size = @sizeOf(T);
    ctx.history.items.len -= size;

    var out: T = undefined;
    const src = ctx.history.items.ptr + ctx.history.items.len;
    @memcpy(std.mem.asBytes(&out), src[0..size]);
    return out;
}
fn getParameters(weights: u32, range: f32) ![]f32 {
    const ctx = context orelse unreachable;
    var params: []f32 = undefined;
    if (ctx.parameterPack.items.len == ctx.parameterCursor) {
        params = try ctx.parameterPack.addManyAsSlice(ctx.gpa, weights);
        switch (ctx.initializer) {
            .uninit => {},
            .uniform => {
                for (params) |*v| {
                    v.* = (ctx.rand.random().float(f32) * 2 - 1) * range;
                }
            },
        }
    } else {
        params = (ctx.parameterPack.items.ptr + ctx.parameterCursor)[0..weights];
    }
    ctx.parameterCursor += weights;
    return params;
}
fn sgd(learningRate: f32) void {
    const ctx = context orelse unreachable;
    for (ctx.parameterPack.items, ctx.parameterGradPack.items) |*param, grad| {
        param.* -= grad * learningRate;
    }
}

// tests

test {
    _ = @import("stridewalker.zig");
}

const TestNet = struct {
    pub const inputLen = 2;

    pub fn forward(input: Tensor) !Tensor {
        var tensor = input;
        tensor = try tensor.linear(3);
        return tensor;
    }
};

const testSeed = 123123123;

fn testingApproxEq(expected: []const f32, actual: []const f32) !void {
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, 0.0001);
    }
}

test "mininet linear" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    var testNet = createNetwork(TestNet);
    @memcpy(testNet.parameters(), &.{ 0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6 });
    try testingApproxEq(&.{ 1.2, 1.9, 2.6 }, try testNet.predict(&.{ 1, 2 }));
    try testingApproxEq(&.{ 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 3, 4 }));
    try testingApproxEq(&.{ 1.2, 1.9, 2.6, 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 1, 2, 3, 4 }));
}

test "mininet backward" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    var testNet = createNetwork(TestNet);
    @memcpy(testNet.parameters(), &.{ 0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6 });

    const scope = TensorScope.save();
    defer scope.restore();

    const xs = try Tensor.init(&.{ 1, 2, 3, 4 }, 2, 2);
    const labels = try Tensor.init(&.{ 1, 2, 3, 4, 5, 6 }, 3, 2);
    const ys = try testNet.predictWithTensor(xs);
    const loss = try ys.l2Loss(labels);
    try loss.backward(&.{1});

    try ys.gradient().testingApproxEq(&.{ 0.4, -0.2, -0.8, -4.4, -3.4, -2.4 });
    try testingApproxEq(&.{ -4.0, -12.8, -16.8, -3.6, -10.4, -14.0, -3.2, -8.0, -11.2 }, testNet.parameterGradients());
    try xs.gradient().testingApproxEq(&.{ -0.42, -0.48, -2.66, -3.68 });
}

test "mininet neg" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -2.0, 0.5, 1.0 }, 3, 1);
    const y = try x.neg();

    try y.testingApproxEq(&.{ 2.0, -0.5, -1.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });
    try x.gradient().testingApproxEq(&.{ -1.0, -1.0, -1.0 });
}

test "mininet add" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x1 = try Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);
    const x2 = try Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);

    const y = try x1.add(x2);

    try y.testingApproxEq(&.{ 11.0, 22.0, 33.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
    try x2.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
}

test "mininet sub" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x1 = try Tensor.init(&.{ 5.0, 6.0, 7.0 }, 3, 1);
    const x2 = try Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);

    const y = try x1.sub(x2);

    try y.testingApproxEq(&.{ 4.0, 4.0, 4.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
    try x2.gradient().testingApproxEq(&.{ -1.0, -1.0, -1.0 });
}

test "mininet mul" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x1 = try Tensor.init(&.{ 2.0, 3.0, 4.0 }, 3, 1);
    const x2 = try Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);

    const y = try x1.mul(x2);

    try y.testingApproxEq(&.{ 20.0, 60.0, 120.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 10.0, 20.0, 30.0 });
    try x2.gradient().testingApproxEq(&.{ 2.0, 3.0, 4.0 });
}

test "mininet div" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x1 = try Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);
    const x2 = try Tensor.init(&.{ 2.0, 4.0, 5.0 }, 3, 1);

    const y = try x1.div(x2);

    try y.testingApproxEq(&.{ 5.0, 5.0, 6.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 0.5, 0.25, 0.2 });
    try x2.gradient().testingApproxEq(&.{ -10.0 / 4.0, -20.0 / 16.0, -30.0 / 25.0 });
}

test "mininet dot" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x1 = try Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);
    const x2 = try Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);

    const y = try x1.dot(x2);

    try y.testingApproxEq(&.{140.0}); // 10 + 40 + 90

    try y.backward(&.{1.0});

    try x1.gradient().testingApproxEq(&.{ 10.0, 20.0, 30.0 });
    try x2.gradient().testingApproxEq(&.{ 1.0, 2.0, 3.0 });
}

test "mininet sum" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);
    const y = try x.sum();

    try y.testingApproxEq(&.{6.0});

    try y.backward(&.{1.0});

    try x.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
}

test "mininet mean" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ 2.0, 4.0, 6.0 }, 3, 1);
    const y = try x.mean();

    try y.testingApproxEq(&.{4.0});

    try y.backward(&.{1.0});

    try x.gradient().testingApproxEq(&.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 });
}

test "mininet relu" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -2.0, -0.5, 0.0, 1.5, 3.0 }, 5, 1);
    const y = try x.relu();

    try y.testingApproxEq(&.{ 0.0, 0.0, 0.0, 1.5, 3.0 });

    try y.backward(&.{ 0.3, -1.2, 2.0, 0.8, -0.4 });

    try x.gradient().testingApproxEq(&.{ 0.0, 0.0, 0.0, 0.8, -0.4 });
}

test "mininet leakyrelu" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const alpha: f32 = 0.1;

    const x = try Tensor.init(&.{ -2.0, -1.0, 0.0, 1.0, 3.0 }, 5, 1);
    const y = try x.leakyRelu(alpha);

    try y.testingApproxEq(&.{
        -0.2, // -2 * 0.1
        -0.1, // -1 * 0.1
        0.0, // 0
        1.0, // x
        3.0, // x
    });

    try y.backward(&.{ 0.5, 1.0, -1.0, 0.3, -0.7 });

    try x.gradient().testingApproxEq(&.{
        0.05, // 0.5 * 0.1
        0.1, // 1.0 * 0.1
        -0.1, // -1.0 * 0.1
        0.3, // 그대로
        -0.7, // 그대로
    });
}

test "mininet sigmoid" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -2.0, 0.0, 2.0 }, 3, 1);
    const y = try x.sigmoid();

    // approximate values
    try y.testingApproxEq(&.{
        0.1192029,
        0.5,
        0.8807971,
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    const ydata = try y.plainData();

    try x.gradient().testingApproxEq(&.{
        ydata[0] * (1 - ydata[0]),
        ydata[1] * (1 - ydata[1]),
        ydata[2] * (1 - ydata[2]),
    });
}

test "mininet tanh" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -2.0, 0.0, 2.0 }, 3, 1);
    const y = try x.tanh();

    try y.testingApproxEq(&.{
        -0.9640276,
        0.0,
        0.9640276,
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    const ydata = try y.plainData();

    try x.gradient().testingApproxEq(&.{
        1 - ydata[0] * ydata[0],
        1 - ydata[1] * ydata[1],
        1 - ydata[2] * ydata[2],
    });
}
