const std = @import("std");
const mininet = @import("mininet");
const xornet = @import("load_raw_generated.zig");

fn network(a: i32, b: i32) i32 {
    const FI2 = mininet.fixed.FITensor(2);
    var net = mininet.fixed.FINetPtr.init(&xornet.parameters);

    // same network structure in load_raw.zig
    return FI2.init(.{ a, b })
        .linear(&net, 4)
        .relu()
        .linear(&net, 1)
        .value[0];
}

fn fixedToF32(value: i32) f32 {
    const fvalue: f32 = @floatFromInt(value);
    return fvalue / mininet.fixed.one;
}

pub fn main(_: std.process.Init) !void {
    const one = mininet.fixed.one;
    std.debug.print("0 xor 0 = {}\n", .{fixedToF32(network(0, 0))});
    std.debug.print("1 xor 0 = {}\n", .{fixedToF32(network(one, 0))});
    std.debug.print("0 xor 1 = {}\n", .{fixedToF32(network(0, one))});
    std.debug.print("1 xor 1 = {}\n", .{fixedToF32(network(one, one))});
}
