const std = @import("std");
const mininet = @import("mininet");

pub fn main(init: std.process.Init) !void {
    // initialize context
    var ctx = mininet.Context.init(init.gpa, 24672672645);
    _ = ctx.set();
    defer ctx.deinit();

    // define network
    var simpleNet = try mininet.createNetwork(struct {
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

    // load
    const parameters = try std.Io.Dir.cwd().readFileAlloc(init.io, "examples/xor.bin", init.gpa, .unlimited);
    @memcpy(std.mem.sliceAsBytes(simpleNet.parameters()), parameters);
    init.gpa.free(parameters);

    // save
    // try std.Io.Dir.cwd().writeFile(init.io, .{
    //     .data = std.mem.sliceAsBytes(simpleNet.parameters()),
    //     .sub_path = "examples/xor.bin",
    // });

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
