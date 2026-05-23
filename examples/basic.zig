const std = @import("std");
const mininet = @import("mininet");

pub fn main(init: std.process.Init) !void {
    // initialize context
    var ctx = try mininet.Context.init(init.gpa);
    _ = ctx.set();
    defer ctx.deinit();

    // define network
    var simpleNet = try ctx.initNetwork(struct {
        pub const inputLen = 2;
        pub fn forward(_: *@This(), input: mininet.Tensor) !mininet.Tensor {
            var tensor = input;
            tensor = try tensor.linear(4);
            tensor = try tensor.relu();
            tensor = try tensor.linear(1);
            return tensor;
        }
    }, .{
        .initializeSeed = 24672672645,
    });
    defer simpleNet.deinit();

    // train
    var opt = mininet.Optimizer.Adam.init(.{ .learningRate = 0.005 });
    try simpleNet.train(&.{ .{
        .input = &.{ 0.0, 0.0 },
        .label = &.{0.0},
    }, .{
        .input = &.{ 1.0, 0.0 },
        .label = &.{1.0},
    }, .{
        .input = &.{ 0.0, 1.0 },
        .label = &.{1.0},
    }, .{
        .input = &.{ 1.0, 1.0 },
        .label = &.{0.0},
    } }, .{
        .optimizer = opt.optimizer(), // default sgd
        .epoch = 500, // default 1000
        .lossFn = mininet.Tensor.l2Loss, // default L2 loss
        .io = init.io, // for printing loss
    });

    // save
    try std.Io.Dir.cwd().writeFile(init.io, .{
        .data = std.mem.sliceAsBytes(simpleNet.parameters()),
        .sub_path = "examples/load.bin",
    });

    // predict
    {
        const res = try simpleNet.predict(&.{ 0, 0 });
        std.debug.print("{}\n", .{res[0]});
    }
    {
        const res = try simpleNet.predict(&.{ 0, 1 });
        std.debug.print("{}\n", .{res[0]});
    }
    {
        const res = try simpleNet.predict(&.{ 1, 0 });
        std.debug.print("{}\n", .{res[0]});
    }
    {
        const res = try simpleNet.predict(&.{ 1, 1 });
        std.debug.print("{}\n", .{res[0]});
    }
}
