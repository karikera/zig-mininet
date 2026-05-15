const std = @import("std");

threadlocal var context: ?*Context = null;

pub const Context = struct {
    gpa: std.mem.Allocator,
    parameterPack: std.ArrayList(f32),
    parameterGradPack: std.ArrayList(f32),
    parameterCursor: u32,
    inputGradPack: std.ArrayList(f32),
    inputPack: std.ArrayList(f32),
    history: std.ArrayList(u8),
    initializer: Initializer,
    rand: std.Random.DefaultPrng,

    pub fn init(gpa: std.mem.Allocator, seed: u64) Context {
        return .{
            .rand = .init(seed), // random seed
            .gpa = gpa,
            .parameterPack = .empty,
            .parameterGradPack = .empty,
            .parameterCursor = 0,
            .inputGradPack = .empty,
            .inputPack = .empty,
            .history = .empty,
            .initializer = .uniform,
        };
    }
    pub fn deinit(ctx: *Context) void {
        if (context == ctx) {
            context = null;
        }
        ctx.parameterPack.deinit(ctx.gpa);
        ctx.parameterGradPack.deinit(ctx.gpa);
        ctx.inputPack.deinit(ctx.gpa);
        ctx.inputGradPack.deinit(ctx.gpa);
        ctx.history.deinit(ctx.gpa);
        ctx.* = undefined;
    }
    pub fn set(ctx: ?*Context) void {
        context = ctx;
    }

    fn gradBytesOffset(ctx: *Context) usize {
        return @intFromPtr(ctx.inputGradPack.items.ptr) -% @intFromPtr(ctx.inputPack.items.ptr);
    }
    fn createTensor(ctx: *Context, values: []const f32, dataLen: u32, batchLen: u32) !Tensor {
        std.debug.assert(dataLen * batchLen == values.len);
        const dataIdx = ctx.inputPack.items.len;
        try ctx.inputPack.appendSlice(ctx.gpa, values);
        return .{ .dataIdx = @intCast(dataIdx), .dataLen = dataLen, .batchStride = dataLen, .batchLen = batchLen };
    }
    fn createDirtyTensor(ctx: *Context, dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        const totalLen = dataLen * batchLen;
        const dataIdx = ctx.inputPack.items.len;
        const array = try ctx.inputPack.addManyAsSlice(ctx.gpa, totalLen);
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

const BackwardFn = *const fn () anyerror!void;

pub const TensorScope = struct {
    inputBegin: usize,

    pub fn save() TensorScope {
        const ctx = context orelse unreachable;
        return .{ .inputBegin = ctx.inputPack.items.len };
    }
    pub fn restore(s: TensorScope) void {
        const ctx = context orelse unreachable;
        std.debug.assert(ctx.inputPack.items.len >= s.inputBegin);
        ctx.inputPack.items.len = s.inputBegin;
    }
};

pub const DataPair = struct {
    input: []const f32,
    label: []const f32,
};

pub fn createNetwork(T: type) Network(T) {
    return Network(T).init();
}

pub fn Network(T: type) type {
    return struct {
        parameterBegin: u32,
        parameterLen: u32,
        implement: *const fn (xs: Tensor) anyerror!Tensor,
        data: T,

        pub fn init() @This() {
            return .{ .parameterBegin = std.math.maxInt(u32), .parameterLen = 0, .implement = T.forward, .data = .{} };
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

            const first = n.parameterBegin == std.math.maxInt(u32);
            if (first) {
                n.parameterBegin = @intCast(ctx.parameterPack.items.len);
            }
            ctx.parameterCursor = n.parameterBegin;
            ctx.history.clearRetainingCapacity();

            const ys = try n.implement(xs);
            const parameterLen: u32 = @intCast(ctx.parameterPack.items.len - n.parameterBegin);
            if (first) {
                n.parameterLen = @intCast(ctx.parameterPack.items.len - n.parameterBegin);
            }
            std.debug.assert(parameterLen == n.parameterLen);
            return ys;
        }

        // return loss
        pub fn trainOnceWithTensor(n: *@This(), xs: Tensor, labels: Tensor, learningRate: f32) !Tensor {
            const ys = try n.predictWithTensor(xs);
            const loss = try ys.l2Loss(labels);
            try loss.backward(&.{1.0});
            Tensor.sgd(learningRate);
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

        pub fn load(n: *@This(), values: []const f32) !void {
            const ctx = context orelse unreachable;
            const oldInit = ctx.initializer;
            defer ctx.initializer = oldInit;
            ctx.initializer = .uninit;

            const scope = TensorScope.save();
            defer scope.restore();

            const xs, _ = try ctx.createDirtyTensor(T.inputLen, 1);
            _ = try n.predictWithTensor(xs);
            if (n.parameterLen != values.len) {
                return Error.SizeMismatch;
            }
            @memcpy((ctx.parameterPack.items.ptr + n.parameterBegin)[0..n.parameterLen], values);
        }
        pub fn parameters(n: *@This()) []f32 {
            const ctx = context orelse unreachable;
            std.debug.assert(n.parameterBegin != std.math.maxInt(u32)); // not initialized
            return ctx.parameterPack.items[n.parameterBegin .. n.parameterBegin + n.parameterLen];
        }
        pub fn parameterGradients(n: *@This()) []f32 {
            const ctx = context orelse unreachable;
            std.debug.assert(n.parameterBegin != std.math.maxInt(u32)); // empty network
            std.debug.assert(ctx.parameterGradPack.items.len >= n.parameterBegin + n.parameterLen); // backward required
            return ctx.parameterGradPack.items[n.parameterBegin .. n.parameterBegin + n.parameterLen];
        }
    };
}

pub const Error = error{ SizeMismatch, BackwardRequired, NotInitialized };

pub const Tensor = struct {
    dataIdx: u32,
    dataLen: u32,
    batchStride: u32,
    batchLen: u32,

    fn u32count(T: type) usize {
        return @divTrunc(@sizeOf(T) + @alignOf(u32) - 1, @alignOf(u32));
    }
    fn writeHistory(value: anytype) !void {
        const T = @TypeOf(value);

        const Impl = struct {
            fn call() !void {
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
        return ctx.inputPack.items.ptr + t.dataIdx;
    }
    fn assertGradient(t: Tensor) void {
        const ctx = context orelse unreachable;
        std.debug.assert(ctx.inputGradPack.items.len >= t.dataIdx + t.dataLen); // backward required
    }
    pub fn gradientPtr(t: Tensor) [*]f32 {
        t.assertGradient();
        const ctx = context orelse unreachable;
        return ctx.inputGradPack.items.ptr + t.dataIdx;
    }
    pub fn dataWalker(t: Tensor) StrideWalker {
        return .init(t.dataPtr(), t.dataLen, t.batchStride, t.batchLen);
    }
    pub fn gradientWalker(t: Tensor) StrideWalker {
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
        t.tensorData().copyTo(dest.ptr);
        return dest;
    }
    pub fn tensorData(t: Tensor) TensorData {
        return .{ .data = t.dataPtr(), .dataLen = t.dataLen, .batchStride = t.batchStride, .batchLen = t.batchLen };
    }
    pub fn tensorGradient(t: Tensor) TensorData {
        t.assertGradient();
        return .{ .data = t.gradientPtr(), .dataLen = t.dataLen, .batchStride = t.batchStride, .batchLen = t.batchLen };
    }
    pub fn backward(t: Tensor, gradient: []const f32) !void {
        const ctx = context orelse unreachable;
        if (t.dataLen * t.batchLen != gradient.len) {
            // only one loss value supported
            return Error.SizeMismatch;
        }

        try ctx.inputGradPack.resize(ctx.gpa, ctx.inputPack.items.len);
        try ctx.parameterGradPack.resize(ctx.gpa, ctx.parameterPack.items.len);

        @memset(ctx.inputGradPack.items, 0.0);
        @memset(ctx.parameterGradPack.items, 0.0);

        var dest = t.tensorGradient();
        dest.copyFrom(gradient.ptr);

        while (ctx.history.items.len > 0) {
            const backFn = readHistory(BackwardFn);
            try backFn();
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
    pub fn linear(t: Tensor, yLen: u32) !Tensor {
        const ctx = context orelse unreachable;
        const xLen = t.dataLen;
        const xStride = t.batchStride;
        const weights = yLen * (xLen + 1);
        const inputLenF32: f32 = @floatFromInt(xLen);
        const outputLenF32: f32 = @floatFromInt(yLen);
        const paramIdx = ctx.parameterCursor;
        const ps: []f32 = try getParameters(weights, std.math.sqrt(6.0 / (inputLenF32 + outputLenF32)));

        const xs = t.dataPtr();
        const outTensor, const ys = try Tensor.initUndef(yLen, t.batchLen);

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
        try writeHistory(LinearBackward{ .t = t, .outIdx = outTensor.dataIdx, .outLen = outTensor.dataLen, .paramIdx = paramIdx });
        return outTensor;
    }
    const LinearBackward = struct {
        t: Tensor,
        outIdx: u32,
        outLen: u32,
        paramIdx: u32,

        fn backward(b: @This()) !void {
            const ctx = context orelse unreachable;
            const xs = b.t.dataPtr();
            const gxs = b.t.gradientPtr();
            const xLen = b.t.dataLen;
            const paramCols = xLen + 1;
            const xStride = b.t.batchStride;
            const xNext = xStride - xLen;
            const yLen = b.outLen;
            const batchLen = b.t.batchLen;
            const gys = ctx.inputGradPack.items.ptr + b.outIdx;
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
                        gpPtr[0] = w;
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
                        gpPtr[0] = w;
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
        try writeHistory(L2LossBackward{ .xt = xt, .labelT = labelT, .outIdx = outTensor.dataIdx });
        return outTensor;
    }
    const L2LossBackward = struct {
        xt: Tensor,
        labelT: Tensor,
        outIdx: u32,

        fn backward(b: @This()) !void {
            const ctx = context orelse unreachable;
            const gy2 = ctx.inputGradPack.items.ptr[b.outIdx] * 2;
            var xWalker = b.xt.dataWalker();
            var labelWalker = b.labelT.dataWalker();
            const gdo = ctx.gradBytesOffset();

            while (xWalker.next()) {
                std.debug.assert(labelWalker.next());
                const grad = (xWalker.ptr[0] - labelWalker.ptr[0]) * gy2;
                xWalker.offset(gdo)[0] += grad;
                labelWalker.offset(gdo)[0] -= grad;
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
            .outIdx = out.dataIdx,
        });
        return out;
    }
    const ReluBackward = struct {
        xt: Tensor,
        outIdx: u32,

        fn backward(b: @This()) !void {
            const ctx = context orelse unreachable;
            var gyPtr = ctx.inputGradPack.items.ptr + b.outIdx;
            var xWalker = b.xt.dataWalker();
            const gbo = ctx.gradBytesOffset();

            while (xWalker.next()) {
                if (xWalker.ptr[0] > 0) {
                    xWalker.offset(gbo)[0] += gyPtr[0];
                }
                gyPtr += 1;
            }
        }
    };
    pub fn sgd(learningRate: f32) void {
        const ctx = context orelse unreachable;
        for (ctx.parameterPack.items, ctx.parameterGradPack.items) |*param, grad| {
            param.* -= grad * learningRate;
        }
    }
};

const StrideWalker = struct {
    ptr: [*]f32,
    end: [*]f32,
    rowEnd: [*]f32,
    widthMinusOne: u32,
    paddingPlusOne: u32,

    fn init(ptr: [*]f32, width: u32, stride: u32, height: u32) StrideWalker {
        const widthMinusOne = width - 1;
        const paddingPlusOne = stride - widthMinusOne;
        return .{
            .ptr = ptr - 1,
            .end = ptr + stride * height - paddingPlusOne,
            .rowEnd = ptr + widthMinusOne,
            .widthMinusOne = widthMinusOne,
            .paddingPlusOne = paddingPlusOne,
        };
    }

    fn next(w: *StrideWalker) bool {
        if (@intFromPtr(w.ptr) < @intFromPtr(w.rowEnd)) {
            w.ptr += 1;
            return true;
        }
        if (@intFromPtr(w.ptr) < @intFromPtr(w.end)) {
            w.ptr += w.paddingPlusOne;
            w.rowEnd = w.ptr + w.widthMinusOne;
            return true;
        }
        return false;
    }

    fn offset(w: *StrideWalker, bytes: usize) [*]f32 {
        return @ptrFromInt(@intFromPtr(w.ptr) +% bytes);
    }

    fn testingApproxEq(w: *StrideWalker, expected: []const f32) !void {
        for (expected) |v| {
            try std.testing.expect(w.next());
            try std.testing.expectApproxEqAbs(v, w.ptr[0], 0.0001);
        }
        try std.testing.expect(!w.next());
    }
};

test "stridewalker" {
    var buf = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var walker = StrideWalker.init(&buf, 3, 4, 2);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(0.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(1.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(2.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(4.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(5.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(6.0, walker.ptr[0]);
    try std.testing.expect(!walker.next());
}

const TestNet = struct {
    pub const inputLen = 2;

    pub fn forward(input: Tensor) !Tensor {
        var tensor = input;
        tensor = try tensor.linear(3);
        return tensor;
    }
};

test "ai linear" {
    var ctx = Context.init(std.testing.allocator);
    Context.set(&ctx);
    defer ctx.deinit();

    var testNet = createNetwork(TestNet);
    try testNet.load(&.{
        0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6,
    });

    try testingApproxEq(&.{ 1.2, 1.9, 2.6 }, try testNet.predict(&.{ 1, 2 }));
    try testingApproxEq(&.{ 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 3, 4 }));
    try testingApproxEq(&.{ 1.2, 1.9, 2.6, 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 1, 2, 3, 4 }));
}

test "ai backward" {
    var ctx = Context.init(std.testing.allocator);
    Context.set(&ctx);
    defer ctx.deinit();

    var testNet = createNetwork(TestNet);
    try testNet.load(&.{
        0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6,
    });
    {
        const scope = TensorScope.save();
        defer scope.restore();

        const xs = try Tensor.init(&.{ 1, 2, 3, 4 }, 2, 2);
        const labels = try Tensor.init(&.{ 1, 2, 3, 4, 5, 6 }, 3, 2);
        const ys = try testNet.predictWithTensor(xs);
        const loss = try ys.l2Loss(labels);
        try loss.backward(&.{1});

        var gyWalker = ys.gradientWalker();
        try gyWalker.testingApproxEq(&.{ 0.4, -0.2, -0.8, -4.4, -3.4, -2.4 });
        try testingApproxEq(&.{ -4.0, -12.8, -16.8, -3.6, -10.4, -14.0, -3.2, -8.0, -11.2 }, testNet.parameterGradients());
        var gxWalker = xs.gradientWalker();
        try gxWalker.testingApproxEq(&.{ -0.42, -0.48, -2.66, -3.68 });
    }
}

test "ai relu" {
    var ctx = Context.init(std.testing.allocator);
    Context.set(&ctx);
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -2.0, -0.5, 0.0, 1.5, 3.0 }, 5, 1);
    const y = try x.relu();
    try y.tensorData().testingApproxEq(&.{ 0.0, 0.0, 0.0, 1.5, 3.0 });
    try y.backward(&.{ 0.3, -1.2, 2.0, 0.8, -0.4 });
    try x.tensorGradient().testingApproxEq(&.{ 0.0, 0.0, 0.0, 0.8, -0.4 });
}

fn testingApproxEq(expected: []const f32, actual: []const f32) !void {
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, 0.0001);
    }
}
