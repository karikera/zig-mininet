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
    cachePack: std.ArrayList(f32),
    history: std.ArrayList(u8),
    rand: std.Random.DefaultPrng,
    initRequested: bool,
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
            .cachePack = .empty,
            .history = .empty,
            .initRequested = false,
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
        ctx.cachePack.deinit(ctx.gpa);
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
    fn createDirtyCache(ctx: *Context, len: u32) !struct { u32, []f32 } {
        const cacheIdx: u32 = @intCast(ctx.cachePack.items.len);
        return .{ cacheIdx, try ctx.cachePack.addManyAsSlice(ctx.gpa, len) };
    }
    fn createDirtyTensor(ctx: *Context, dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        const totalLen = dataLen * batchLen;
        const dataIdx = ctx.dataPack.items.len;
        return .{
            .{ .dataIdx = @intCast(dataIdx), .dataLen = dataLen, .batchStride = dataLen, .batchLen = batchLen },
            try ctx.dataPack.addManyAsSlice(ctx.gpa, totalLen),
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
    fn getParameters(ctx: *Context, weights: u32) ![]f32 {
        var params: []f32 = undefined;
        if (ctx.parameterPack.items.len == ctx.parameterCursor) {
            params = try ctx.parameterPack.addManyAsSlice(ctx.gpa, weights);
        } else {
            params = (ctx.parameterPack.items.ptr + ctx.parameterCursor)[0..weights];
        }
        ctx.parameterCursor += weights;
        return params;
    }

    pub fn initNetwork(ctx: *Context, T: type) !Network(T) {
        const scope = TensorScope.save();
        defer scope.restore();

        const xs, _ = try ctx.createDirtyTensor(T.inputLen, 1);
        var n: Network(T) = .{
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
    dataIdx: usize,
    cacheIdx: usize,

    pub fn save() TensorScope {
        const ctx = context orelse unreachable;
        return .{ .dataIdx = ctx.dataPack.items.len, .cacheIdx = ctx.cachePack.items.len };
    }
    pub fn restore(s: TensorScope) void {
        const ctx = context orelse unreachable;
        std.debug.assert(ctx.dataPack.items.len >= s.dataIdx);
        std.debug.assert(ctx.cachePack.items.len >= s.cacheIdx);
        ctx.dataPack.items.len = s.dataIdx;
        ctx.cachePack.items.len = s.cacheIdx;
    }
};

pub const DataPair = struct {
    input: []const f32,
    label: []const f32,
};

pub const Optimizer = struct {
    optimizeFn: *const fn (data: *anyopaque) void,
    data: *anyopaque,

    pub const SGD = struct {
        learningRate: f32,
        parameterBegin: u32,
        parameterLen: u32,

        pub fn init(net: anytype, learningRate: f32) !SGD {
            return .{
                .learningRate = learningRate,
                .parameterBegin = net.parameterBegin,
                .parameterLen = net.parameterLen,
            };
        }

        pub fn optimizer(this: *SGD) Optimizer {
            const Impl = struct {
                fn optimizeImpl(data: *anyopaque) void {
                    const sgd: *SGD = @ptrCast(@alignCast(data));
                    const ctx = context orelse unreachable;
                    const params = ctx.parameterPack.items[sgd.parameterBegin .. sgd.parameterBegin + sgd.parameterLen];
                    const grads = ctx.parameterGradPack.items[sgd.parameterBegin .. sgd.parameterBegin + sgd.parameterLen];
                    const lr = sgd.learningRate;
                    for (params, grads) |*param, grad| {
                        param.* -= grad * lr;
                    }
                }
            };
            return .{
                .data = @ptrCast(this),
                .optimizeFn = Impl.optimizeImpl,
            };
        }
    };
    pub const Adam = struct {
        learningRate: f32,
        b1: f32,
        b2: f32,
        epsilon: f32,
        parameterBegin: u32,
        parameterLen: u32,
        data: []f32,
        step: f32,

        pub const Options = struct {
            learningRate: f32 = 0.001,
            b1: f32 = 0.9,
            b2: f32 = 0.999,
            epsilon: f32 = 1e-8,
        };

        pub fn init(net: anytype, opts: Options) !Adam {
            const ctx = context orelse unreachable;
            const data = try ctx.gpa.alloc(f32, net.parameterLen * 2);
            @memset(data, 0.0);

            return .{
                .learningRate = opts.learningRate,
                .b1 = opts.b1,
                .b2 = opts.b2,
                .epsilon = opts.epsilon,
                .parameterBegin = net.parameterBegin,
                .parameterLen = net.parameterLen,
                .data = data,
                .step = 0,
            };
        }
        pub fn deinit(adam: *Adam) void {
            const ctx = context orelse unreachable;
            ctx.gpa.free(adam.data);
            adam.* = undefined;
        }

        pub fn optimizer(this: *Adam) Optimizer {
            const Impl = struct {
                fn optimizeImpl(data: *anyopaque) void {
                    const adam: *Adam = @ptrCast(@alignCast(data));
                    const ctx = context orelse unreachable;

                    var mvPtr = adam.data.ptr;
                    const b1 = adam.b1;
                    const b2 = adam.b2;
                    const ib1 = 1 - adam.b1;
                    const ib2 = 1 - adam.b2;
                    const params = ctx.parameterPack.items[adam.parameterBegin .. adam.parameterBegin + adam.parameterLen];
                    const grads = ctx.parameterGradPack.items[adam.parameterBegin .. adam.parameterBegin + adam.parameterLen];
                    const lr = adam.learningRate;
                    const e = adam.epsilon;
                    adam.step += 1.0;
                    const ibt1 = 1 - std.math.pow(f32, b1, adam.step);
                    const ibt2 = 1 - std.math.pow(f32, b2, adam.step);

                    for (params, grads) |*param, grad| {
                        var m = mvPtr[0];
                        m = b1 * m + ib1 * grad;
                        mvPtr[0] = m;
                        mvPtr += 1;
                        var v = mvPtr[0];
                        v = b2 * v + ib2 * (grad * grad);
                        mvPtr[0] = v;
                        mvPtr += 1;
                        const res = lr * (m / ibt1) / (std.math.sqrt(v / ibt2) + e);
                        param.* -= res;
                    }
                }
            };
            return .{
                .data = @ptrCast(this),
                .optimizeFn = Impl.optimizeImpl,
            };
        }
    };

    pub fn optimize(opt: Optimizer) void {
        opt.optimizeFn(opt.data);
    }
};

pub fn Network(T: type) type {
    return struct {
        parameterBegin: u32,
        parameterLen: u32,
        implement: *const fn (xs: Tensor) anyerror!Tensor,
        data: T,

        pub fn deinit(n: *@This()) void {
            n.* = undefined;
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
        pub fn trainOnceWithTensor(n: *@This(), xs: Tensor, labels: Tensor, optimizer: Optimizer) !Tensor {
            const scope = TensorScope.save();
            defer scope.restore();
            const ys = try n.predictWithTensor(xs);
            const loss = try ys.l2Loss(labels);
            try loss.backward(&.{1.0});
            optimizer.optimize();
            return loss;
        }

        pub fn train(n: *@This(), data: []const DataPair, epoch: u32, optimizer: Optimizer) !void {
            const ctx = context orelse unreachable;
            if (data.len == 0) return;

            const scope = TensorScope.save();
            defer scope.restore();

            const xs = try ctx.generateInputTensor(data);
            const labels = try ctx.generateLabelTensor(data);

            var lossSum: f32 = 0;
            var printedStep: u32 = 0;
            var step: u32 = 0;
            const sumMax = @max((epoch + 5) / 10, 1);

            for (0..epoch) |_| {
                const loss = try n.trainOnceWithTensor(xs, labels, optimizer);
                lossSum += loss.dataPtr()[0];
                step += 1;
                const sumCount = step - printedStep;
                if (sumCount >= sumMax) {
                    const sumCountF: f32 = @floatFromInt(sumCount);
                    printedStep = step;

                    const lossMean = lossSum / sumCountF;
                    lossSum = 0;
                    std.debug.print("[{}] Loss = {}\n", .{ step, lossMean });
                }
            }

            const sumCount = step - printedStep;
            if (sumCount > 0) {
                const sumCountF: f32 = @floatFromInt(sumCount);
                const lossMean = lossSum / sumCountF;
                std.debug.print("[{}] Loss = {}\n", .{ step, lossMean });
            }
        }

        pub fn randomInitialize(n: *@This()) !void {
            const ctx = context orelse unreachable;
            ctx.initRequested = true;
            defer ctx.initRequested = false;

            const scope = TensorScope.save();
            defer scope.restore();

            const xs, _ = try ctx.createDirtyTensor(T.inputLen, 1);
            _ = try n.predictWithTensor(xs);
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
    pub fn format(
        t: Tensor,
        writer: *std.Io.Writer,
    ) !void {
        return t.data().format(writer);
    }
    fn testingApproxEq(t: Tensor, expected: []const f32) !void {
        return t.data().testingApproxEq(expected);
    }

    pub fn linear(xt: Tensor, yLen: u32) !Tensor {
        const ctx = context orelse unreachable;
        const xLen = xt.dataLen;
        const xStride = xt.batchStride;
        const weights = yLen * (xLen + 1);
        const paramIdx = ctx.parameterCursor;
        const ps: []f32 = try ctx.getParameters(weights);
        if (ctx.initRequested) {
            const inputLenF32: f32 = @floatFromInt(xLen);
            const outputLenF32: f32 = @floatFromInt(yLen);
            const range = std.math.sqrt(6.0 / (inputLenF32 + outputLenF32));
            for (ps) |*v| {
                v.* = (ctx.rand.random().float(f32) * 2 - 1) * range;
            }
        }

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
            var xWalker = b.xt.dataWalker();
            var labelWalker = b.labelT.dataWalker();
            const gdo = ctx.gradDataOffset();
            const total: f32 = @floatFromInt(b.xt.dataLen * b.xt.batchLen);
            const gy = ctx.dataGradPack.items.ptr[b.yIdx];
            const c = gy * 2 / total;

            while (xWalker.next()) {
                std.debug.assert(labelWalker.next());
                const grad = (xWalker.ptr[0] - labelWalker.ptr[0]) * c;
                gdo.ptr(xWalker.ptr)[0] += grad;
                gdo.ptr(labelWalker.ptr)[0] -= grad;
            }
        }
    };

    // mono

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
    pub fn exp(xt: Tensor) !Tensor {
        var xWalker = xt.dataWalker();
        const yt, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);
        var yPtr = ys.ptr;
        while (xWalker.next()) {
            yPtr[0] = std.math.exp(xWalker.ptr[0]);
            yPtr += 1;
        }
        try writeHistory(ExpBackward{
            .xt = xt,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const ExpBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gxWalker = b.xt.gradientWalker();
            var yPtr = ctx.dataPack.items.ptr + b.yIdx;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (gxWalker.next()) {
                gxWalker.ptr[0] += yPtr[0] * gyPtr[0];
                yPtr += 1;
                gyPtr += 1;
            }
        }
    };

    // bi

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
    pub fn pow(x1t: Tensor, x2t: Tensor) !Tensor {
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try Tensor.initUndef(x1t.dataLen, x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = std.math.pow(f32, x1Walker.ptr[0], x2Walker.ptr[0]);
            yPtr += 1;
        }
        try writeHistory(PowBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yIdx = yt.dataIdx,
        });
        return yt;
    }
    const PowBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var x1Walker = b.x1t.dataWalker();
            var x2Walker = b.x2t.dataWalker();
            const gdo = ctx.gradDataOffset();
            var yPtr = ctx.dataPack.items.ptr + b.yIdx;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            while (x1Walker.next()) {
                _ = x2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                const x1 = x1Walker.ptr[0];
                const x2 = x2Walker.ptr[0];
                const y = yPtr[0];
                yPtr += 1;

                gdo.ptr(x1Walker.ptr)[0] += y * x2 / x1 * gy;
                gdo.ptr(x2Walker.ptr)[0] += y * @log(x1) * gy;
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

    // activations
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
    pub fn elu(xt: Tensor, alpha: f32) !Tensor {
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        while (xWalker.next()) {
            if (xWalker.ptr[0] > 0) {
                yPtr[0] = xWalker.ptr[0];
            } else {
                yPtr[0] = (std.math.exp(xWalker.ptr[0]) - 1) * alpha;
            }
            yPtr += 1;
        }
        try writeHistory(EluBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
            .alpha = alpha,
        });
        return out;
    }
    const EluBackward = struct {
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
                const x = xWalker.ptr[0];
                gbo.ptr(xWalker.ptr)[0] += if (x > 0) gy else std.math.exp(x) * gy * b.alpha;
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
                gxWalker.ptr[0] += gyPtr[0] * y * (1 - y);
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
                gxWalker.ptr[0] += gyPtr[0] * (1 - y * y);
                yPtr += 1;
                gyPtr += 1;
            }
        }
    };
    pub fn softmax(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        const scope = TensorScope.save();
        defer scope.restore();

        _, const xsExp = try ctx.createDirtyCache(xt.dataLen);

        var sumVal: f32 = 0;
        var xWalker = xt.dataWalkerRow();
        var yPtr = ys.ptr;
        while (xWalker.next()) |xRow| {
            var xExpPtr = xsExp.ptr;
            for (xRow) |x| {
                const xExp = std.math.exp(x);
                sumVal += xExp;
                xExpPtr[0] = xExp;
                xExpPtr += 1;
            }
            for (xsExp) |xExp| {
                yPtr[0] = xExp / sumVal;
                yPtr += 1;
            }
        }

        try writeHistory(SoftmaxBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
        });
        return out;
    }
    const SoftmaxBackward = struct {
        xt: Tensor,
        yIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            const ys = ctx.dataPack.items.ptr + b.yIdx;

            const gdo = ctx.gradDataOffset();
            var yPtr = ys;

            var gxWalker = b.xt.gradientWalkerRow();
            while (gxWalker.next()) |gxRow| {
                const yRowBegin = yPtr;
                const yRowEnd = yPtr + b.xt.dataLen;
                var sumVal: f32 = 0.0;
                while (@intFromPtr(yPtr) < @intFromPtr(yRowEnd)) {
                    sumVal += gdo.ptr(yPtr)[0] * yPtr[0];
                    yPtr += 1;
                }
                yPtr = yRowBegin;
                for (gxRow) |*gx| {
                    gx.* = (gdo.ptr(yPtr)[0] - sumVal) * yPtr[0];
                    yPtr += 1;
                }
            }
        }
    };
    const geluConstant: f32 = std.math.sqrt(2.0 / std.math.pi);
    pub fn gelu(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        const out, const ys = try Tensor.initUndef(xt.dataLen, xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        const cacheIdx, const cache = try ctx.createDirtyCache(xt.dataLen * xt.batchLen);

        var cachePtr = cache.ptr;
        while (xWalker.next()) {
            const x = xWalker.ptr[0];
            const t = std.math.tanh(geluConstant * (x + 0.044715 * x * x * x));
            cachePtr[0] = t;
            yPtr[0] = 0.5 * x * (1 + t);
            yPtr += 1;
            cachePtr += 1;
        }
        try writeHistory(GeluBackward{
            .xt = xt,
            .yIdx = out.dataIdx,
            .cacheIdx = cacheIdx,
        });
        return out;
    }
    const GeluBackward = struct {
        xt: Tensor,
        yIdx: u32,
        cacheIdx: u32,

        fn backward(b: @This()) void {
            const ctx = context orelse unreachable;
            var gyPtr = ctx.dataGradPack.items.ptr + b.yIdx;
            var cachePtr = ctx.cachePack.items.ptr + b.cacheIdx;
            var xWalker = b.xt.dataWalker();
            const gbo = ctx.gradDataOffset();

            while (xWalker.next()) {
                const x = xWalker.ptr[0];
                const t = cachePtr[0];
                gbo.ptr(xWalker.ptr)[0] +=
                    0.5 * (1.0 + t) +
                    0.5 * x * (1.0 - t * t) *
                        geluConstant * (1.0 + 0.134145 * x * x);
                gyPtr += 1;
                cachePtr += 1;
            }
        }
    };
    pub fn silu(xt: Tensor) !Tensor {
        const sig = try xt.sigmoid();
        return sig.mul(xt);
    }

    // TODO: log, clamp, abs, sqrt

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

    var testNet = try ctx.initNetwork(TestNet);
    @memcpy(testNet.parameters(), &[_]f32{ 0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6 });
    try testingApproxEq(&.{ 1.2, 1.9, 2.6 }, try testNet.predict(&.{ 1, 2 }));
    try testingApproxEq(&.{ 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 3, 4 }));
    try testingApproxEq(&.{ 1.2, 1.9, 2.6, 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 1, 2, 3, 4 }));
}

test "mininet backward" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    var testNet = try ctx.initNetwork(TestNet);
    @memcpy(testNet.parameters(), &[_]f32{ 0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6 });

    const scope = TensorScope.save();
    defer scope.restore();

    const xs = try Tensor.init(&.{ 1, 2, 3, 4 }, 2, 2);
    const labels = try Tensor.init(&.{ 1, 2, 3, 4, 5, 6 }, 3, 2);
    const ys = try testNet.predictWithTensor(xs);
    const loss = try ys.l2Loss(labels);
    try loss.backward(&.{1});

    try ys.gradient().testingApproxEq(&.{ 0.2 / 3.0, -0.1 / 3.0, -0.4 / 3.0, -2.2 / 3.0, -1.7 / 3.0, -0.4 });
    try testingApproxEq(&.{ -2.0 / 3.0, -6.4 / 3.0, -2.8, -0.6, -5.2 / 3.0, -7.0 / 3.0, -1.6 / 3.0, -4.0 / 3.0, -5.6 / 3.0 }, testNet.parameterGradients());
    try xs.gradient().testingApproxEq(&.{ -0.07, -0.08, -1.33 / 3.0, -1.84 / 3.0 });
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

test "mininet exp" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ 0.0, 1.0, 2.0 }, 3, 1);

    const y = try x.exp();

    try y.testingApproxEq(&.{
        1.0,
        std.math.e,
        std.math.exp(2.0),
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x.gradient().testingApproxEq(&.{
        1.0,
        std.math.e,
        std.math.exp(2.0),
    });
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

test "mininet pow" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x1 = try Tensor.init(&.{ 2.0, 3.0, 4.0 }, 3, 1);
    const x2 = try Tensor.init(&.{ 3.0, 2.0, 0.5 }, 3, 1);

    const y = try x1.pow(x2);

    try y.testingApproxEq(&.{
        8.0,
        9.0,
        2.0,
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{
        12.0,
        6.0,
        0.25,
    });

    try x2.gradient().testingApproxEq(&.{
        8.0 * @log(2.0),
        9.0 * @log(3.0),
        2.0 * @log(4.0),
    });
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

test "mininet elu" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -1.0, 0.0, 1.0 }, 3, 1);

    // alpha = 1.0
    const y = try x.elu(1.0);

    try y.testingApproxEq(&.{
        -0.63212056, // exp(-1) - 1
        0.0,
        1.0,
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    // derivative:
    // x <= 0 : exp(x)
    // x > 0  : 1
    try x.gradient().testingApproxEq(&.{
        0.36787945, // exp(-1)
        1.0,
        1.0,
    });
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

test "mininet softmax" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);

    const y = try x.softmax();

    // exp(x) / sum(exp(x))
    try y.testingApproxEq(&.{
        0.09003057,
        0.24472848,
        0.66524094,
    });

    try y.backward(&.{ 1.0, 2.0, 3.0 });

    // softmax backward:
    // dX = Y * (dY - sum(dY * Y))
    //
    // dot =
    // 1*y0 + 2*y1 + 3*y2
    // = 2.57521031

    try x.gradient().testingApproxEq(&.{
        -0.1418171,
        -0.1407704,
        0.2825875,
    });
}

test "mininet gelu" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -1.0, 0.0, 1.0 }, 3, 1);

    const y = try x.gelu();

    try y.testingApproxEq(&.{
        -0.158808,
        0.0,
        0.841192,
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    // approximate GELU derivative
    try x.gradient().testingApproxEq(&.{
        -0.082964,
        0.5,
        1.082964,
    });
}

test "mininet silu" {
    var ctx = Context.init(std.testing.allocator, testSeed);
    _ = ctx.set();
    defer ctx.deinit();

    const scope = TensorScope.save();
    defer scope.restore();

    const x = try Tensor.init(&.{ -1.0, 0.0, 1.0 }, 3, 1);

    const y = try x.silu();

    try y.testingApproxEq(&.{
        -0.26894143,
        0.0,
        0.7310586,
    });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x.gradient().testingApproxEq(&.{
        0.07232949,
        0.5,
        0.92767054,
    });
}
