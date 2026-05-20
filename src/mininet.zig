const std = @import("std");
const StrideWalker = @import("stridewalker.zig");

threadlocal var context: ?*Context = null;

const TensorStoreData = struct {
    data: std.ArrayList(f32),
    grad: std.ArrayList(f32),
    train: std.ArrayList(f32),

    pub const empty: TensorStoreData = .{
        .data = .empty,
        .grad = .empty,
        .train = .empty,
    };

    pub fn deinit(store: *TensorStoreData, gpa: std.mem.Allocator) void {
        store.data.deinit(gpa);
        store.grad.deinit(gpa);
        store.train.deinit(gpa);
        store.* = undefined;
    }

    fn gradDataOffset(store: *TensorStoreData) GradDataOffset {
        return .{ .bytes = @intFromPtr(store.grad.items.ptr) -% @intFromPtr(store.data.items.ptr) };
    }
};

const TensorPointer = struct {
    store: TensorStore,
    dataIdx: u32,

    pub fn dataPtr(tp: TensorPointer) [*]f32 {
        return tp.store.data().data.items.ptr + tp.dataIdx;
    }
    pub fn gradientPtr(tp: TensorPointer) [*]f32 {
        return tp.store.data().grad.items.ptr + tp.dataIdx;
    }
    pub fn assertGradient(tp: TensorPointer, len: usize) void {
        std.debug.assert(tp.store.data().grad.items.len >= tp.dataIdx + len); // backward required
    }
};

const DefTensorPointer = struct {
    dataIdx: u32,

    pub fn initUndef(totalLen: u32) !struct { DefTensorPointer, []f32 } {
        const ctx = context orelse unreachable;
        const store = TensorStore.def.data();
        const dataIdx = store.data.items.len;
        return .{
            .{
                .dataIdx = @intCast(dataIdx),
            },
            try store.data.addManyAsSlice(ctx.gpa, totalLen),
        };
    }
    pub fn dataPtr(tp: DefTensorPointer) [*]f32 {
        return TensorStore.def.data().data.items.ptr + tp.dataIdx;
    }
    pub fn gradientPtr(tp: DefTensorPointer) [*]f32 {
        return TensorStore.def.data().grad.items.ptr + tp.dataIdx;
    }
    pub fn assertGradient(tp: DefTensorPointer, len: usize) void {
        std.debug.assert(TensorStore.def.data().grad.items.len >= tp.dataIdx + len); // backward required
    }
    pub fn asTensor(tp: DefTensorPointer, dataLen: u32, batchStride: u32, batchLen: u32) Tensor {
        return .{
            .ptr = .{
                .store = TensorStore.def,
                .dataIdx = tp.dataIdx,
            },
            .dataLen = dataLen,
            .batchStride = batchStride,
            .batchLen = batchLen,
        };
    }
};

pub const Context = struct {
    gpa: std.mem.Allocator,
    rand: std.Random.DefaultPrng,
    stores: std.ArrayList(TensorStoreData),
    parameterCursor: u32,
    parameterStore: TensorStore,
    initRequested: bool,
    usingCheck: std.atomic.Value(bool),
    history: HistoryBuffer,

    pub fn init(gpa: std.mem.Allocator, seed: u64) !Context {
        var stores = try std.ArrayList(TensorStoreData).initCapacity(gpa, 2);
        stores.appendAssumeCapacity(.empty);
        stores.appendAssumeCapacity(.empty);
        return .{
            .gpa = gpa,
            .rand = .init(seed), // random seed
            .stores = stores,
            .parameterCursor = 0,
            .parameterStore = undefined,
            .initRequested = false,
            .usingCheck = .init(false),
            .history = .empty,
        };
    }
    pub fn deinit(ctx: *Context) void {
        if (context == ctx) {
            context = null;
        }
        ctx.history.deinit(ctx.gpa);
        for (ctx.stores.items) |*store| {
            store.deinit(ctx.gpa);
        }
        ctx.stores.deinit(ctx.gpa);
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

    fn getParameters(ctx: *Context, weights: u32) ![]f32 {
        var params: []f32 = undefined;
        const store = &ctx.stores.items[ctx.parameterStore.id];
        if (store.data.items.len == ctx.parameterCursor) {
            params = try store.data.addManyAsSlice(ctx.gpa, weights);
        } else {
            params = (store.data.items.ptr + ctx.parameterCursor)[0..weights];
        }
        ctx.parameterCursor += weights;
        return params;
    }
    fn parameterPtr(ctx: *Context) TensorPointer {
        return .{
            .store = ctx.parameterStore,
            .dataIdx = ctx.parameterCursor,
        };
    }
    fn createRawTensor(ctx: *Context, store: TensorStore, dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        const sd = &ctx.stores.items[store.id];
        const totalLen = dataLen * batchLen;
        const dataIdx = sd.data.items.len;
        return .{
            .{
                .ptr = .{
                    .store = store,
                    .dataIdx = @intCast(dataIdx),
                },
                .dataLen = dataLen,
                .batchStride = dataLen,
                .batchLen = batchLen,
            },
            try sd.data.addManyAsSlice(ctx.gpa, totalLen),
        };
    }

    pub fn initNetwork(ctx: *Context, T: type, opts: NetworkOptions) !Network(T) {
        return .init(ctx, opts);
    }
    pub fn initStore(ctx: *Context) !TensorStore {
        const id = ctx.stores.items.len;
        try ctx.stores.append(ctx.gpa, .empty);
        return .{ .id = @intCast(id) };
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

    pub fn batch(t: TensorData, batchIndex: u32) []f32 {
        const off = batchIndex * t.batchStride;
        return t.data[off .. off + t.dataLen];
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
    pub fn testingApproxEq(t: TensorData, expected: []const f32) !void {
        var walker = StrideWalker.init(t.data, t.dataLen, t.batchStride, t.batchLen);
        return walker.testingApproxEq(expected);
    }
};

const Initializer = enum {
    uninit,
    uniform,
};

pub const TensorStore = struct {
    id: u32,

    pub const def: TensorStore = .{ .id = 0 };
    pub const network: TensorStore = .{ .id = 1 };

    pub fn data(ts: TensorStore) *TensorStoreData {
        const ctx = context orelse unreachable;
        return &ctx.stores.items[ts.id];
    }

    pub fn create(store: TensorStore, values: []f32, dataLen: u32, batchLen: u32) !Tensor {
        const out, const dest = try store.createUndef(dataLen, batchLen);
        @memcpy(dest, values);
        return out;
    }
    pub fn createUndef(store: TensorStore, dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        const ctx = context orelse unreachable;
        return ctx.createRawTensor(store, dataLen, batchLen);
    }
    pub fn collectInputsOf(store: TensorStore, dataPair: []const DataPair) !Tensor {
        std.debug.assert(dataPair.len != 0);
        const inputLen: u32 = @intCast(dataPair[0].input.len);
        const t, const dest = try store.createUndef(inputLen, @intCast(dataPair.len));
        var destPtr = dest.ptr;
        for (dataPair) |d| {
            @memcpy(destPtr[0..inputLen], d.input);
            destPtr += inputLen;
        }
        return t;
    }
    pub fn collectLabelsOf(store: TensorStore, dataPair: []const DataPair) !Tensor {
        std.debug.assert(dataPair.len != 0);
        const outputLen: u32 = @intCast(dataPair[0].label.len);
        const t, const dest = try store.createUndef(outputLen, @intCast(dataPair.len));
        var destPtr = dest.ptr;
        for (dataPair) |d| {
            @memcpy(destPtr[0..outputLen], d.label);
            destPtr += outputLen;
        }
        return t;
    }
};

pub const TensorScope = struct {
    dataIdx: usize,

    pub fn save() TensorScope {
        return .{
            .dataIdx = TensorStore.def.data().data.items.len,
        };
    }
    pub fn restore(s: TensorScope) void {
        const sd = TensorStore.def.data();
        std.debug.assert(sd.data.items.len >= s.dataIdx);
        sd.data.items.len = s.dataIdx;
    }
};

pub const DataPair = struct {
    input: []const f32,
    label: []const f32,
};

pub const Optimizer = struct {
    optimizeFn: *const fn (data: *anyopaque, stores: []const TensorStore) std.mem.Allocator.Error!void,
    data: *anyopaque,

    pub const SGD = struct {
        learningRate: f32,

        pub fn init(learningRate: f32) !SGD {
            return .{
                .learningRate = learningRate,
            };
        }

        pub fn deinit(this: *SGD) void {
            this.* = undefined;
        }

        pub fn optimizer(this: *SGD) Optimizer {
            const Impl = struct {
                fn optimizeImpl(data: *anyopaque, stores: []const TensorStore) std.mem.Allocator.Error!void {
                    const sgd: *SGD = @ptrCast(@alignCast(data));
                    const lr = sgd.learningRate;
                    for (stores) |store| {
                        const sd = store.data();
                        const params = sd.data.items;
                        const grads = sd.grad.items;
                        for (params, grads) |*param, grad| {
                            param.* -= grad * lr;
                        }
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
        step: f32,

        pub const Options = struct {
            learningRate: f32 = 0.001,
            b1: f32 = 0.9,
            b2: f32 = 0.999,
            epsilon: f32 = 1e-8,
        };

        pub fn init(opts: Options) !Adam {
            return .{
                .learningRate = opts.learningRate,
                .b1 = opts.b1,
                .b2 = opts.b2,
                .epsilon = opts.epsilon,
                .step = 0,
            };
        }
        pub fn deinit(this: *Adam) void {
            this.* = undefined;
        }

        pub fn optimizer(this: *Adam) Optimizer {
            const Impl = struct {
                fn optimizeImpl(data: *anyopaque, stores: []const TensorStore) std.mem.Allocator.Error!void {
                    const ctx = context orelse unreachable;
                    const adam: *Adam = @ptrCast(@alignCast(data));

                    const b1 = adam.b1;
                    const b2 = adam.b2;
                    const ib1 = 1 - adam.b1;
                    const ib2 = 1 - adam.b2;
                    const lr = adam.learningRate;
                    const e = adam.epsilon;
                    adam.step += 1.0;
                    const ibt1 = 1 - std.math.pow(f32, b1, adam.step);
                    const ibt2 = 1 - std.math.pow(f32, b2, adam.step);

                    for (stores) |store| {
                        const sd = store.data();
                        const parameterLen = sd.data.items.len;
                        if (sd.train.items.len == 0) {
                            try sd.train.resize(ctx.gpa, parameterLen * 2);
                            @memset(sd.train.items, 0.0);
                        } else {
                            std.debug.assert(sd.train.items.len == parameterLen * 2); // parameter length changed
                        }
                        var mvPtr = sd.train.items.ptr;
                        const params = sd.data.items;
                        const grads = sd.grad.items;

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
                }
            };
            return .{
                .data = @ptrCast(this),
                .optimizeFn = Impl.optimizeImpl,
            };
        }
    };

    pub fn optimize(opt: Optimizer, stores: []const TensorStore) !void {
        return opt.optimizeFn(opt.data, stores);
    }
};

pub const NetworkOptions = struct {
    // dirty state if it's true
    noInitialize: bool = false,

    store: TensorStore = TensorStore.network,
};

pub fn Network(T: type) type {
    return struct {
        parameter: TensorPointer,
        parameterLen: u32,
        implement: *const fn (xs: Tensor) std.mem.Allocator.Error!Tensor,
        data: T,

        fn init(ctx: *Context, opts: NetworkOptions) !@This() {
            const scope = TensorScope.save();
            defer scope.restore();

            const xs, _ = try ctx.createRawTensor(TensorStore.def, T.inputLen, 1);
            const store = opts.store;
            const sd = store.data();
            const parameterIdx: u32 = @intCast(sd.data.items.len);
            ctx.parameterCursor = parameterIdx;
            const oldStore = ctx.parameterStore;
            defer ctx.parameterStore = oldStore;
            ctx.parameterStore = store;
            ctx.history.clearRetainingCapacity();

            if (!opts.noInitialize) ctx.initRequested = true;
            defer ctx.initRequested = false;
            _ = try T.forward(xs);
            const parameterLen: u32 = @intCast(sd.data.items.len - parameterIdx);

            return .{
                .parameter = .{
                    .store = store,
                    .dataIdx = parameterIdx,
                },
                .parameterLen = parameterLen,
                .implement = T.forward,
                .data = .{},
            };
        }

        pub fn deinit(n: *@This()) void {
            n.* = undefined;
        }

        pub fn predict(n: *@This(), input: []const f32) ![]const f32 {
            if (input.len == 0) {
                return &.{};
            }

            const scope = TensorScope.save();
            defer scope.restore();

            const batchLen: u32 = @intCast(input.len / T.inputLen);
            const xs, const dest = try TensorStore.def.createUndef(T.inputLen, batchLen);
            @memcpy(dest, input);
            const ys = try n.predictWithTensor(xs);
            return ys.plainData();
        }

        pub fn predictWithTensor(n: *@This(), xs: Tensor) !Tensor {
            const ctx = context orelse unreachable;
            if (xs.dataLen != T.inputLen) return Error.SizeMismatch;

            ctx.parameterCursor = n.parameter.dataIdx;
            const oldStore = ctx.parameterStore;
            defer ctx.parameterStore = oldStore;
            ctx.parameterStore = n.parameter.store;
            ctx.history.clearRetainingCapacity();

            const ys = try n.implement(xs);
            const parameterLen: u32 = @intCast(n.parameter.store.data().data.items.len - n.parameter.dataIdx);
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
            try optimizer.optimize(&.{
                n.parameter.store,
            });
            return loss;
        }

        pub fn train(n: *@This(), data: []const DataPair, epoch: u32, optimizer: Optimizer) !void {
            if (data.len == 0) return;

            const scope = TensorScope.save();
            defer scope.restore();

            const xs = try Tensor.collectInputsOf(data);
            const labels = try Tensor.collectLabelsOf(data);

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

        pub fn parameters(n: *@This()) []f32 {
            return n.parameter.dataPtr()[0..n.parameterLen];
        }
        pub fn parameterGradients(n: *@This()) []f32 {
            n.parameter.assertGradient(n.parameterLen);
            return n.parameter.gradientPtr()[0..n.parameterLen];
        }
    };
}

pub const Error = error{SizeMismatch};

pub const Tensor = struct {
    ptr: TensorPointer,
    dataLen: u32,
    batchStride: u32,
    batchLen: u32,

    pub fn init(values: []const f32, dataLen: u32, batchLen: u32) !Tensor {
        const out, const dest = try TensorStore.def.createUndef(dataLen, batchLen);
        @memcpy(dest, values);
        return out;
    }
    pub fn initUndef(dataLen: u32, batchLen: u32) !struct { Tensor, []f32 } {
        return TensorStore.def.createUndef(dataLen, batchLen);
    }
    pub fn dataPtr(t: Tensor) [*]f32 {
        return t.ptr.dataPtr();
    }
    fn assertGradient(t: Tensor) void {
        t.ptr.assertGradient(t.batchStride * t.batchLen);
    }
    pub fn collectLabelsOf(dataPair: []const DataPair) !Tensor {
        return TensorStore.def.collectLabelsOf(dataPair);
    }
    pub fn collectInputsOf(dataPair: []const DataPair) !Tensor {
        return TensorStore.def.collectInputsOf(dataPair);
    }
    pub fn gradientPtr(t: Tensor) [*]f32 {
        t.assertGradient();
        return t.ptr.gradientPtr();
    }
    pub fn gradDataOffset(t: Tensor) GradDataOffset {
        return t.ptr.store.data().gradDataOffset();
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

    // it possibly returns a temporal slice
    pub fn plainData(t: Tensor) ![]const f32 {
        const ctx = context orelse unreachable;
        if (t.batchLen == 1) {
            return t.dataPtr()[0..t.dataLen];
        }
        if (t.batchStride == t.dataLen) {
            return t.dataPtr()[0 .. t.batchStride * t.batchLen];
        }

        const dest = try ctx.history.useTemporalCache(ctx.gpa, t.dataLen * t.batchLen);
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

        for (ctx.stores.items) |*store| {
            try store.grad.resize(ctx.gpa, store.data.items.len);
            @memset(store.grad.items, 0.0);
        }

        var dest = t.gradient();
        dest.copyFrom(grad.ptr);
        ctx.history.execute();
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
    pub fn testingApproxEq(t: Tensor, expected: []const f32) !void {
        return t.data().testingApproxEq(expected);
    }

    pub fn linear(xt: Tensor, yLen: u32) !Tensor {
        const ctx = context orelse unreachable;
        const xLen = xt.dataLen;
        const xStride = xt.batchStride;
        const weights = yLen * (xLen + 1);

        const param = ctx.parameterPtr();
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
        const yt, const ys = try DefTensorPointer.initUndef(yLen * xt.batchLen);

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
        try ctx.history.write(ctx.gpa, LinearBackward{
            .xt = xt,
            .yt = yt,
            .yLen = yLen,
            .param = param,
        });
        return yt.asTensor(yLen, yLen, xt.batchLen);
    }
    const LinearBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,
        yLen: u32,
        param: TensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            const xs = b.xt.dataPtr();
            const gxs = b.xt.gradientPtr();
            const xLen = b.xt.dataLen;
            const paramCols = xLen + 1;
            const xStride = b.xt.batchStride;
            const xNext = xStride - xLen;
            const yLen = b.yLen;
            const batchLen = b.xt.batchLen;
            const gys = b.yt.gradientPtr();
            const gyEnd = gys + yLen * batchLen;

            var gyPtr = gys;

            // weight/bias grads
            {
                const gps = b.param.gradientPtr();
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
            const ps = b.param.dataPtr();
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
        const ctx = context orelse unreachable;
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

        const yt, const ys = try DefTensorPointer.initUndef(1);
        ys[0] = out;
        try ctx.history.write(ctx.gpa, L2LossBackward{ .xt = xt, .labelT = labelT, .yt = yt });
        return yt.asTensor(1, 1, 1);
    }
    const L2LossBackward = struct {
        xt: Tensor,
        labelT: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var xWalker = b.xt.dataWalker();
            var labelWalker = b.labelT.dataWalker();
            const xGdo = b.xt.ptr.store.data().gradDataOffset();
            const labelGdo = b.labelT.ptr.store.data().gradDataOffset();
            const total: f32 = @floatFromInt(b.xt.dataLen * b.xt.batchLen);
            const gy = b.yt.gradientPtr()[0];
            const c = gy * 2 / total;

            while (xWalker.next()) {
                std.debug.assert(labelWalker.next());
                const grad = (xWalker.ptr[0] - labelWalker.ptr[0]) * c;
                xGdo.ptr(xWalker.ptr)[0] += grad;
                labelGdo.ptr(labelWalker.ptr)[0] -= grad;
            }
        }
    };

    // mono

    pub fn neg(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        var xWalker = xt.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);
        var yPtr = ys.ptr;
        while (xWalker.next()) {
            yPtr[0] = -xWalker.ptr[0];
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, NegBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const NegBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gxWalker = b.xt.gradientWalker();
            var gyPtr = b.yt.gradientPtr();
            while (gxWalker.next()) {
                const gy = gyPtr[0];
                gyPtr += 1;
                gxWalker.ptr[0] -= gy;
            }
        }
    };
    pub fn exp(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        var xWalker = xt.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);
        var yPtr = ys.ptr;
        while (xWalker.next()) {
            yPtr[0] = std.math.exp(xWalker.ptr[0]);
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, ExpBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const ExpBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gxWalker = b.xt.gradientWalker();
            var yPtr = b.yt.dataPtr();
            var gyPtr = b.yt.gradientPtr();
            while (gxWalker.next()) {
                gxWalker.ptr[0] += yPtr[0] * gyPtr[0];
                yPtr += 1;
                gyPtr += 1;
            }
        }
    };

    // bi

    pub fn add(x1t: Tensor, x2t: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(x1t.dataLen * x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] + x2Walker.ptr[0];
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, AddBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yt = yt,
        });
        return yt.asTensor(x1t.dataLen, x1t.dataLen, x1t.batchLen);
    }
    const AddBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gx1Walker = b.x1t.gradientWalker();
            var gx2Walker = b.x2t.gradientWalker();
            var gyPtr = b.yt.gradientPtr();
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
        const ctx = context orelse unreachable;
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(x1t.dataLen * x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] - x2Walker.ptr[0];
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, SubBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yt = yt,
        });
        return yt.asTensor(x1t.dataLen, x1t.dataLen, x1t.batchLen);
    }
    const SubBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gx1Walker = b.x1t.gradientWalker();
            var gx2Walker = b.x2t.gradientWalker();
            var gyPtr = b.yt.gradientPtr();
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
        const ctx = context orelse unreachable;
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(x1t.dataLen * x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] * x2Walker.ptr[0];
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, MulBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yt = yt,
        });
        return yt.asTensor(x1t.dataLen, x1t.dataLen, x1t.batchLen);
    }
    const MulBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var x1Walker = b.x1t.dataWalker();
            var x2Walker = b.x2t.dataWalker();
            const x1Gdo = b.x1t.gradDataOffset();
            const x2Gdo = b.x2t.gradDataOffset();
            var gyPtr = b.yt.gradientPtr();
            while (x1Walker.next()) {
                _ = x2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;
                x1Gdo.ptr(x1Walker.ptr)[0] += x2Walker.ptr[0] * gy;
                x2Gdo.ptr(x2Walker.ptr)[0] += x1Walker.ptr[0] * gy;
            }
        }
    };
    pub fn div(x1t: Tensor, x2t: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(x1t.dataLen * x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = x1Walker.ptr[0] / x2Walker.ptr[0];
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, DivBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yt = yt,
        });
        return yt.asTensor(x1t.dataLen, x1t.dataLen, x1t.batchLen);
    }
    const DivBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var x1Walker = b.x1t.dataWalker();
            var x2Walker = b.x2t.dataWalker();
            const x1Gdo = b.x1t.gradDataOffset();
            const x2Gdo = b.x2t.gradDataOffset();
            var gyPtr = b.yt.gradientPtr();
            while (x1Walker.next()) {
                _ = x2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                const x2 = x2Walker.ptr[0];
                x1Gdo.ptr(x1Walker.ptr)[0] += gy / x2;
                x2Gdo.ptr(x2Walker.ptr)[0] += gy * -x1Walker.ptr[0] / (x2 * x2);
            }
        }
    };
    pub fn pow(x1t: Tensor, x2t: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalker();
        var x2Walker = x2t.dataWalker();
        const yt, const ys = try DefTensorPointer.initUndef(x1t.dataLen * x1t.batchLen);
        var yPtr = ys.ptr;
        while (x1Walker.next()) {
            _ = x2Walker.next();
            yPtr[0] = std.math.pow(f32, x1Walker.ptr[0], x2Walker.ptr[0]);
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, PowBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yt = yt,
        });
        return yt.asTensor(x1t.dataLen, x1t.dataLen, x1t.batchLen);
    }
    const PowBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var x1Walker = b.x1t.dataWalker();
            var x2Walker = b.x2t.dataWalker();
            const x1Gdo = b.x1t.gradDataOffset();
            const x2Gdo = b.x2t.gradDataOffset();
            var yPtr = b.yt.dataPtr();
            var gyPtr = b.yt.gradientPtr();
            while (x1Walker.next()) {
                _ = x2Walker.next();
                const gy = gyPtr[0];
                gyPtr += 1;

                const x1 = x1Walker.ptr[0];
                const x2 = x2Walker.ptr[0];
                const y = yPtr[0];
                yPtr += 1;

                x1Gdo.ptr(x1Walker.ptr)[0] += y * x2 / x1 * gy;
                x2Gdo.ptr(x2Walker.ptr)[0] += y * @log(x1) * gy;
            }
        }
    };
    pub fn dot(x1t: Tensor, x2t: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        if (x1t.dataLen != x2t.dataLen) return Error.SizeMismatch;
        if (x1t.batchLen != x2t.batchLen) return Error.SizeMismatch;
        var x1Walker = x1t.dataWalkerRow();
        var x2Walker = x2t.dataWalkerRow();
        const yt, const ys = try DefTensorPointer.initUndef(x1t.batchLen);

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

        try ctx.history.write(ctx.gpa, DotBackward{
            .x1t = x1t,
            .x2t = x2t,
            .yt = yt,
        });
        return yt.asTensor(1, 1, x1t.batchLen);
    }
    const DotBackward = struct {
        x1t: Tensor,
        x2t: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var x1Walker = b.x1t.dataWalkerRow();
            var x2Walker = b.x2t.dataWalkerRow();
            const x1Gdo = b.x1t.gradDataOffset();
            const x2Gdo = b.x2t.gradDataOffset();
            var gyPtr = b.yt.gradientPtr();
            while (x1Walker.next()) |x1row| {
                const x2row = x2Walker.next().?;

                const gy = gyPtr[0];
                gyPtr += 1;

                for (x1row, x2row, x1Gdo.slice(x1row), x2Gdo.slice(x2row)) |x1, x2, *gx1, *gx2| {
                    gx1.* += gy * x2;
                    gx2.* += gy * x1;
                }
            }
        }
    };
    pub fn sum(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        var xWalker = xt.dataWalkerRow();
        const yt, const ys = try DefTensorPointer.initUndef(xt.batchLen);

        var yPtr = ys.ptr;
        while (xWalker.next()) |xrow| {
            var sumVal: f32 = 0;
            for (xrow) |x1| {
                sumVal += x1;
            }
            yPtr[0] = sumVal;
            yPtr += 1;
        }

        try ctx.history.write(ctx.gpa, SumBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(1, 1, xt.batchLen);
    }
    const SumBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gxWalker = b.xt.gradientWalkerRow();
            var gyPtr = b.yt.gradientPtr();
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
        const ctx = context orelse unreachable;
        var xWalker = xt.dataWalkerRow();
        const yt, const ys = try DefTensorPointer.initUndef(xt.batchLen);
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

        try ctx.history.write(ctx.gpa, MeanBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(1, 1, xt.batchLen);
    }
    const MeanBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            const xLenF32: f32 = @floatFromInt(b.xt.dataLen);
            var gxWalker = b.xt.gradientWalkerRow();
            var gyPtr = b.yt.gradientPtr();
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
        const ctx = context orelse unreachable;
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

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
        try ctx.history.write(ctx.gpa, ReluBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const ReluBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gyPtr = b.yt.gradientPtr();
            var xWalker = b.xt.dataWalker();
            const xGdo = b.xt.gradDataOffset();

            while (xWalker.next()) {
                if (xWalker.ptr[0] > 0) {
                    xGdo.ptr(xWalker.ptr)[0] += gyPtr[0];
                }
                gyPtr += 1;
            }
        }
    };
    pub fn leakyRelu(xt: Tensor, alpha: f32) !Tensor {
        const ctx = context orelse unreachable;
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

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
        try ctx.history.write(ctx.gpa, LeakyReluBackward{
            .xt = xt,
            .yt = yt,
            .alpha = alpha,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const LeakyReluBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,
        alpha: f32,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gyPtr = b.yt.gradientPtr();
            var xWalker = b.xt.dataWalker();
            const xGdo = b.xt.gradDataOffset();

            while (xWalker.next()) {
                const gy = gyPtr[0];
                xGdo.ptr(xWalker.ptr)[0] += if (xWalker.ptr[0] > 0) gy else gy * b.alpha;
                gyPtr += 1;
            }
        }
    };
    pub fn elu(xt: Tensor, alpha: f32) !Tensor {
        const ctx = context orelse unreachable;
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

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
        try ctx.history.write(ctx.gpa, EluBackward{
            .xt = xt,
            .yt = yt,
            .alpha = alpha,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const EluBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,
        alpha: f32,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var gyPtr = b.yt.gradientPtr();
            var xWalker = b.xt.dataWalker();
            const xGdo = b.xt.gradDataOffset();

            while (xWalker.next()) {
                const gy = gyPtr[0];
                const x = xWalker.ptr[0];
                xGdo.ptr(xWalker.ptr)[0] += if (x > 0) gy else std.math.exp(x) * gy * b.alpha;
                gyPtr += 1;
            }
        }
    };
    pub fn sigmoid(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

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
        try ctx.history.write(ctx.gpa, SigmoidBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const SigmoidBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var yPtr = b.yt.dataPtr();
            var gyPtr = b.yt.gradientPtr();
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
        const ctx = context orelse unreachable;
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        while (xWalker.next()) {
            yPtr[0] = std.math.tanh(xWalker.ptr[0]);
            yPtr += 1;
        }
        try ctx.history.write(ctx.gpa, TanhBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const TanhBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            var yPtr = b.yt.dataPtr();
            var gyPtr = b.yt.gradientPtr();
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
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

        const scope = TensorScope.save();
        defer scope.restore();

        const xsExp = try ctx.history.useTemporalCache(ctx.gpa, xt.dataLen);

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

        try ctx.history.write(ctx.gpa, SoftmaxBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const SoftmaxBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), _: *HistoryBuffer) void {
            const ys = b.yt.dataPtr();

            const xGdo = b.xt.gradDataOffset();
            const yGdo = TensorStore.def.data().gradDataOffset();
            var yPtr = ys;

            var gxWalker = b.xt.gradientWalkerRow();
            while (gxWalker.next()) |gxRow| {
                const yRowBegin = yPtr;
                const yRowEnd = yPtr + b.xt.dataLen;
                var sumVal: f32 = 0.0;
                while (@intFromPtr(yPtr) < @intFromPtr(yRowEnd)) {
                    sumVal += xGdo.ptr(yPtr)[0] * yPtr[0];
                    yPtr += 1;
                }
                yPtr = yRowBegin;
                for (gxRow) |*gx| {
                    gx.* = (yGdo.ptr(yPtr)[0] - sumVal) * yPtr[0];
                    yPtr += 1;
                }
            }
        }
    };
    const geluConstant: f32 = std.math.sqrt(2.0 / std.math.pi);
    pub fn gelu(xt: Tensor) !Tensor {
        const ctx = context orelse unreachable;
        const yt, const ys = try DefTensorPointer.initUndef(xt.dataLen * xt.batchLen);

        var xWalker = xt.dataWalker();
        var yPtr = ys.ptr;

        const cache = try ctx.history.addManyCache(ctx.gpa, xt.dataLen * xt.batchLen);

        var cachePtr = cache.ptr;
        while (xWalker.next()) {
            const x = xWalker.ptr[0];
            const t = std.math.tanh(geluConstant * (x + 0.044715 * x * x * x));
            cachePtr[0] = t;
            yPtr[0] = 0.5 * x * (1 + t);
            yPtr += 1;
            cachePtr += 1;
        }
        try ctx.history.write(ctx.gpa, GeluBackward{
            .xt = xt,
            .yt = yt,
        });
        return yt.asTensor(xt.dataLen, xt.dataLen, xt.batchLen);
    }
    const GeluBackward = struct {
        xt: Tensor,
        yt: DefTensorPointer,

        fn backward(b: *@This(), history: *HistoryBuffer) void {
            const total = b.xt.dataLen * b.xt.batchLen;
            var cachePtr = history.readCache(total).ptr;
            var gyPtr = b.yt.gradientPtr();
            var xWalker = b.xt.dataWalker();
            const xGdo = b.xt.gradDataOffset();

            while (xWalker.next()) {
                const x = xWalker.ptr[0];
                const t = cachePtr[0];
                xGdo.ptr(xWalker.ptr)[0] +=
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

const HistoryBuffer = struct {
    const Fn = *const fn (h: *HistoryBuffer) void;

    buf: std.ArrayList(u32),

    pub const empty: HistoryBuffer = .{ .buf = .empty };

    fn assertU32Safe(T: type) void {
        if (@alignOf(T) < @alignOf(u32)) {
            @compileError("alignment mismatch");
        }
        if (@sizeOf(T) % @sizeOf(u32) != 0) {
            @compileError("size mismatch");
        }
    }
    fn u32len(T: type) usize {
        return @divExact(@sizeOf(T), @sizeOf(u32));
    }
    fn u32ArrayMut(T: type, data: *T) *[u32len(T)]u32 {
        assertU32Safe(T);
        return @ptrCast(data);
    }
    fn u32Array(T: type, data: *const T) *const [u32len(T)]u32 {
        assertU32Safe(T);
        return @ptrCast(data);
    }

    fn clearRetainingCapacity(history: *HistoryBuffer) void {
        history.buf.clearRetainingCapacity();
    }
    fn deinit(history: *HistoryBuffer, gpa: std.mem.Allocator) void {
        history.buf.deinit(gpa);
        history.* = undefined;
    }
    fn useTemporalCache(history: *HistoryBuffer, gpa: std.mem.Allocator, len: usize) ![]f32 {
        try history.buf.ensureTotalCapacity(gpa, history.buf.items.len + len);
        const ptr: [*]f32 = @ptrCast(history.buf.items.ptr);
        return (ptr + history.buf.items.len)[0..len];
    }
    fn addManyCache(history: *HistoryBuffer, gpa: std.mem.Allocator, len: usize) ![]f32 {
        const out = try history.buf.addManyAsSlice(gpa, len);
        const ptr: [*]f32 = @ptrCast(out.ptr);
        return ptr[0..len];
    }
    fn readCache(history: *HistoryBuffer, len: usize) []f32 {
        history.buf.items.len -= len;
        const out: [*]f32 = @ptrCast(history.buf.items.ptr + history.buf.items.len);
        return out[0..len];
    }
    fn write(history: *HistoryBuffer, gpa: std.mem.Allocator, value: anytype) !void {
        const T = @TypeOf(value);

        const Impl = struct {
            fn call(h: *HistoryBuffer) void {
                if (@alignOf(T) > @alignOf(u32)) {
                    @compileError("invalid alignment");
                }
                const len = u32len(T);
                h.buf.items.len -= len;
                var v: *T = @ptrCast(h.buf.items.ptr + h.buf.items.len);
                return v.backward(h);
            }
        };

        const buf = try history.buf.addManyAsArray(gpa, @divExact(@sizeOf(T) + @sizeOf(Fn), @sizeOf(u32)));
        const slice = u32Array(T, &value);
        @memcpy(buf[0..slice.len], slice);
        const f: Fn = Impl.call;
        @memcpy(buf[slice.len..], u32Array(Fn, &f));
    }
    fn read(history: *HistoryBuffer, T: type) T {
        const size = @divExact(@sizeOf(T), @sizeOf(u32));
        history.buf.items.len -= size;

        var out: T = undefined;
        const src = history.buf.items.ptr + history.buf.items.len;
        @memcpy(u32ArrayMut(T, &out), src[0..size]);
        return out;
    }
    fn execute(history: *HistoryBuffer) void {
        while (history.buf.items.len > 0) {
            const f = history.read(Fn);
            f(history);
        }
    }
};
