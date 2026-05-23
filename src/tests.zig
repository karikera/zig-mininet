const mininet = @import("mininet.zig");
const std = @import("std");

const TestNet = struct {
    pub const inputLen = 2;

    pub fn forward(_: *@This(), input: mininet.Tensor) !mininet.Tensor {
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

test {
    _ = @import("stridewalker.zig");
}

test "mininet linear" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    var testNet = try ctx.initNetwork(TestNet, .{});
    @memcpy(testNet.parameters(), &[_]f32{ 0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6 });
    try testingApproxEq(&.{ 1.2, 1.9, 2.6 }, try testNet.predict(&.{ 1, 2 }));
    try testingApproxEq(&.{ 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 3, 4 }));
    try testingApproxEq(&.{ 1.2, 1.9, 2.6, 1.8, 3.3, 4.8 }, try testNet.predict(&.{ 1, 2, 3, 4 }));
}

test "mininet backward" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    var testNet = try ctx.initNetwork(TestNet, .{});
    @memcpy(testNet.parameters(), &[_]f32{ 0.7, 0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.5, 0.6 });

    const scope = mininet.TensorScope.save();
    defer scope.restore();

    const xs = try mininet.Tensor.init(&.{ 1, 2, 3, 4 }, 2, 2);
    const labels = try mininet.Tensor.init(&.{ 1, 2, 3, 4, 5, 6 }, 3, 2);
    const ys = try testNet.predictWithTensor(xs);
    const loss = try ys.l2Loss(labels);
    try loss.backward(&.{1});

    try ys.gradient().testingApproxEq(&.{ 0.2 / 3.0, -0.1 / 3.0, -0.4 / 3.0, -2.2 / 3.0, -1.7 / 3.0, -0.4 });
    try testingApproxEq(&.{ -2.0 / 3.0, -6.4 / 3.0, -2.8, -0.6, -5.2 / 3.0, -7.0 / 3.0, -1.6 / 3.0, -4.0 / 3.0, -5.6 / 3.0 }, testNet.parameterGradients());
    try xs.gradient().testingApproxEq(&.{ -0.07, -0.08, -1.33 / 3.0, -1.84 / 3.0 });
}

test "mininet optimizer" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    var testNet = try ctx.initNetwork(TestNet, .{
        .initializeSeed = testSeed,
    });
    const pair: []const mininet.DataPair = &.{
        .{ .input = &.{ 1, 2 }, .label = &.{ 1, 2, 3 } },
    };

    try testNet.train(pair, .{});

    var sgd = mininet.Optimizer.SGD.init(0.001);
    try testNet.train(pair, .{
        .optimizer = sgd.optimizer(),
        .lossFn = mininet.Tensor.l1Loss,
    });

    var adam = mininet.Optimizer.Adam.init(.{});
    try testNet.train(pair, .{
        .optimizer = adam.optimizer(),
        .lossFn = mininet.Tensor.l2Loss,
    });
}

test "mininet neg" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -2.0, 0.5, 1.0 }, 3, 1);
    const y = try x.neg();

    try y.testingApproxEq(&.{ 2.0, -0.5, -1.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });
    try x.gradient().testingApproxEq(&.{ -1.0, -1.0, -1.0 });
}

test "mininet exp" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ 0.0, 1.0, 2.0 }, 3, 1);

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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x1 = try mininet.Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);
    const x2 = try mininet.Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);

    const y = try x1.add(x2);

    try y.testingApproxEq(&.{ 11.0, 22.0, 33.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
    try x2.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
}

test "mininet sub" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x1 = try mininet.Tensor.init(&.{ 5.0, 6.0, 7.0 }, 3, 1);
    const x2 = try mininet.Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);

    const y = try x1.sub(x2);

    try y.testingApproxEq(&.{ 4.0, 4.0, 4.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
    try x2.gradient().testingApproxEq(&.{ -1.0, -1.0, -1.0 });
}

test "mininet mul" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x1 = try mininet.Tensor.init(&.{ 2.0, 3.0, 4.0 }, 3, 1);
    const x2 = try mininet.Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);

    const y = try x1.mul(x2);

    try y.testingApproxEq(&.{ 20.0, 60.0, 120.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 10.0, 20.0, 30.0 });
    try x2.gradient().testingApproxEq(&.{ 2.0, 3.0, 4.0 });
}

test "mininet div" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x1 = try mininet.Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);
    const x2 = try mininet.Tensor.init(&.{ 2.0, 4.0, 5.0 }, 3, 1);

    const y = try x1.div(x2);

    try y.testingApproxEq(&.{ 5.0, 5.0, 6.0 });

    try y.backward(&.{ 1.0, 1.0, 1.0 });

    try x1.gradient().testingApproxEq(&.{ 0.5, 0.25, 0.2 });
    try x2.gradient().testingApproxEq(&.{ -10.0 / 4.0, -20.0 / 16.0, -30.0 / 25.0 });
}

test "mininet pow" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x1 = try mininet.Tensor.init(&.{ 2.0, 3.0, 4.0 }, 3, 1);
    const x2 = try mininet.Tensor.init(&.{ 3.0, 2.0, 0.5 }, 3, 1);

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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x1 = try mininet.Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);
    const x2 = try mininet.Tensor.init(&.{ 10.0, 20.0, 30.0 }, 3, 1);

    const y = try x1.dot(x2);

    try y.testingApproxEq(&.{140.0}); // 10 + 40 + 90

    try y.backward(&.{1.0});

    try x1.gradient().testingApproxEq(&.{ 10.0, 20.0, 30.0 });
    try x2.gradient().testingApproxEq(&.{ 1.0, 2.0, 3.0 });
}

test "mininet sum" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);
    const y = try x.sum();

    try y.testingApproxEq(&.{6.0});

    try y.backward(&.{1.0});

    try x.gradient().testingApproxEq(&.{ 1.0, 1.0, 1.0 });
}

test "mininet mean" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ 2.0, 4.0, 6.0 }, 3, 1);
    const y = try x.mean();

    try y.testingApproxEq(&.{4.0});

    try y.backward(&.{1.0});

    try x.gradient().testingApproxEq(&.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 });
}

test "mininet relu" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -2.0, -0.5, 0.0, 1.5, 3.0 }, 5, 1);
    const y = try x.relu();

    try y.testingApproxEq(&.{ 0.0, 0.0, 0.0, 1.5, 3.0 });

    try y.backward(&.{ 0.3, -1.2, 2.0, 0.8, -0.4 });

    try x.gradient().testingApproxEq(&.{ 0.0, 0.0, 0.0, 0.8, -0.4 });
}

test "mininet elu" {
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -1.0, 0.0, 1.0 }, 3, 1);

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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const alpha: f32 = 0.1;

    const x = try mininet.Tensor.init(&.{ -2.0, -1.0, 0.0, 1.0, 3.0 }, 5, 1);
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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -2.0, 0.0, 2.0 }, 3, 1);
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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -2.0, 0.0, 2.0 }, 3, 1);
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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ 1.0, 2.0, 3.0 }, 3, 1);

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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -1.0, 0.0, 1.0 }, 3, 1);

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
    var ctx = try mininet.Context.init(std.testing.allocator);
    _ = ctx.set();
    defer ctx.deinit();

    const x = try mininet.Tensor.init(&.{ -1.0, 0.0, 1.0 }, 3, 1);

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
