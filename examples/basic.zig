const std = @import("std");
const mininet = @import("mininet");

pub fn main(init: std.process.Init) !void {
    // initialize context
    var ctx = mininet.Context.init(init.gpa, 24672672645);
    mininet.Context.set(&ctx);
    defer ctx.deinit();

    // define network
    var simpleNet = mininet.createNetwork(struct {
        pub const inputLen = 2;
        pub fn forward(input: mininet.Tensor) !mininet.Tensor {
            var tensor = input;
            tensor = try tensor.linear(3);
            tensor = try tensor.relu();
            tensor = try tensor.linear(3);
            tensor = try tensor.linear(1);
            return tensor;
        }
    });

    // train
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
    } }, 0.02, 1000);

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
