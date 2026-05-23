const std = @import("std");

// fixed int network for determinism
// non reuse-able
// must be created for each prediction
pub const FINetPtr = struct {
    ptr: [*]const u32,
    end: [*]const u32,

    pub fn init(net: []const u32) FINetPtr {
        const ptr = net.ptr;
        return .{
            .ptr = ptr,
            .end = ptr + net.len,
        };
    }
};

// fixed int tensor for determinism
pub fn FITensor(comptime xLen: usize) type {
    return struct {
        value: [xLen]i32,

        pub fn init(value: [xLen]i32) @This() {
            return .{ .value = value };
        }
        pub fn linear(t: @This(), net: *FINetPtr, comptime yLen: usize) FITensor(yLen) {
            std.debug.assert(@intFromPtr(net.ptr + xLen * yLen) <= @intFromPtr(net.end));
            var p = net.ptr;
            var out: FITensor(yLen) = undefined;
            for (&out.value) |*y| {
                var w: i32 = mulF32(one, p[0]);
                p += 1;

                for (t.value) |x| {
                    w +|= mulF32(x, p[0]);
                    p += 1;
                }
                y.* = w;
            }
            net.ptr = p;
            return out;
        }
        pub fn relu(t: @This()) FITensor(xLen) {
            var out: FITensor(xLen) = undefined;
            for (&out.value, t.value) |*y, x| {
                if (x > 0) {
                    y.* = x;
                } else {
                    y.* = 0;
                }
            }
            return out;
        }
        pub fn silu(t: @This()) FITensor(xLen) {
            var out: FITensor(xLen) = undefined;
            for (&out.value, t.value) |*y, x| {
                var res: i64 = undefined;
                if (x > 0) {
                    const z = exp(-x);
                    res = @as(i64, x) * one / (one + @as(i64, z));
                } else {
                    const z = exp(x);
                    res = @as(i64, x) * @as(i64, z) / (one + @as(i64, z));
                }
                y.* = maxClamp(res);
            }
        }
    };
}

pub const oneShift = 8;
pub const one = 1 << oneShift;
pub const mantissaMask = one - 1;

fn maxClamp(value: i64) i32 {
    if (value < std.math.minInt(i32)) {
        return std.math.minInt(i32);
    } else if (value > std.math.maxInt(i32)) {
        return std.math.maxInt(i32);
    } else {
        return @intCast(value);
    }
}

// saturating multiplication
pub fn mulF32(value: i32, rawF32: u32) i32 {
    if (value == 0) return 0;
    const irawF32: i32 = @bitCast(rawF32);

    const absValue = @abs(value);
    const absRawFloat: u32 = @intCast(irawF32 & 0x7fffffff);

    var fexp = @as(i32, @intCast((absRawFloat >> 23) & 0xff)) - 127;
    const clz = @clz(absValue);
    if (fexp >= clz) {
        // overflow
        if (value ^ irawF32 < 0) {
            return std.math.minInt(i32);
        } else {
            return std.math.maxInt(i32);
        }
    }

    const mantissa = (absRawFloat & 0x7fffff) | 0x800000;
    const multiplied = @as(i64, mantissa) * @as(i64, absValue);
    fexp -= 23;
    var shifted: i64 = undefined;
    if (fexp >= 0) {
        shifted = multiplied << @intCast(fexp);
    } else {
        if (fexp <= -63) {
            return 0;
        }
        fexp = -fexp;
        const half: i64 = (@as(i64, 1) << @intCast(fexp)) >> 1;
        shifted = (multiplied + half) >> @intCast(fexp);
    }
    if (value ^ irawF32 > 0) {
        return @intCast(@min(std.math.maxInt(i32), shifted));
    } else {
        return @intCast(@max(-shifted, std.math.minInt(i32)));
    }
}

// fixedOne based exp function
const expTable = @import("generated_exp_table.zig");

pub fn exp(value: i32) u32 {
    if (oneShift != 8) {
        @compileError("unexpected fixedOneShift");
    }
    const int = value >> oneShift;
    const highM: u32 = @intCast((value & 0xf0) >> 4);
    const lowM: u32 = @intCast(value & 0xf);
    if (int >= expTable.expIntOffset + expTable.expIntTable.len) {
        return std.math.maxInt(u32);
    }
    if (int <= expTable.expMinusIntOffset - @as(i32, expTable.expMinusIntTable.len)) {
        return 0;
    }
    const b = expTable.expHighMTable[highM];
    const c = expTable.expLowMTable[lowM];

    var res: u64 = undefined;
    if (int >= expTable.expIntOffset) {
        const a = expTable.expIntTable[@intCast(int - expTable.expIntOffset)];
        const shift = expTable.expIntShift + expTable.expHighMShift + expTable.expLowMShift - oneShift;
        res = (@as(u64, a) * @as(u64, b) * @as(u64, c) + (1 << (shift - 1))) >> shift;
    } else {
        const a = expTable.expMinusIntTable[@intCast(expTable.expMinusIntOffset - int)];
        const shift = expTable.expMinusIntShift + expTable.expHighMShift + expTable.expLowMShift - oneShift;
        res = (@as(u64, a) * @as(u64, b) * @as(u64, c) + (1 << (shift - 1))) >> shift;
    }
    if (res > std.math.maxInt(u32)) {
        return std.math.maxInt(u32);
    }
    return @intCast(res);
}

test "fixed mulf32" {
    const one_f: f32 = 1.0;
    const zero_one_f: f32 = 0.1;
    const one_one_f: f32 = 1.1;
    const e: f32 = 0.000001;
    try std.testing.expectEqual(10, mulF32(100, @bitCast(zero_one_f)));
    try std.testing.expectEqual(1, mulF32(10, @bitCast(zero_one_f)));
    try std.testing.expectEqual(0x7fffffff, mulF32(0x7fffffff, @bitCast(one_f)));
    try std.testing.expectEqual(-0x80000000, mulF32(-0x80000000, @bitCast(one_f)));
    try std.testing.expectEqual(-0x7fffffff, mulF32(-0x7fffffff, @bitCast(one_f)));
    try std.testing.expectEqual(-0x80000000, mulF32(-0x7fffffff, @bitCast(one_one_f)));
    try std.testing.expectEqual(0x7fffffff, mulF32(0x7ffffffe, @bitCast(one_one_f)));
    try std.testing.expectEqual(11, mulF32(10, @bitCast(one_one_f)));
    try std.testing.expectEqual(0, mulF32(1, @bitCast(e)));
    try std.testing.expectEqual(0, mulF32(0, 0));
}

test "fixed exp" {
    // 1. e^0 = 1.0 (256)
    try std.testing.expectEqual(@as(u32, 256), exp(0));

    // 2. 소수점 이하 최소값: e^(1/256) ≈ 1.0039 (257)
    try std.testing.expectEqual(@as(u32, 257), exp(1));

    // 3. e^1.0 = 2.718... (696)
    // 1.0은 8비트 시프트된 256입니다.
    try std.testing.expectEqual(@as(u32, 696), exp(256));

    // 4. e^2.0 = 7.389... (1892)
    // 2.0은 512입니다.
    try std.testing.expectEqual(@as(u32, 1892), exp(512));

    // 5. 오버플로우 상한선 체크 (정수부 17 이상)
    // 17.0 = 17 << 8 = 4352
    try std.testing.expectEqual(std.math.maxInt(u32), exp(4352));
    try std.testing.expectEqual(std.math.maxInt(u32), exp(std.math.maxInt(i32)));
}

test "fixed fullexp" {
    // 1. e^0.5 검증 (정수=0, HighM=8, LowM=0)
    // 수학적 실제값: e^0.5 ≈ 1.64872 -> 고정소수점 변환: 1.64872 * 256 ≈ 422
    // 입력값: 0.5 * 256 = 128 (바이너리: 00000000 1000 0000)
    try std.testing.expectEqual(@as(u32, 422), exp(128));

    // 2. e^1.25 검증 (정수=1, HighM=4, LowM=0)
    // 수학적 실제값: e^1.25 ≈ 3.49034 -> 고정소수점 변환: 3.49034 * 256 ≈ 893
    // 입력값: 1.25 * 256 = 320 (바이너리: 00000001 0100 0000)
    try std.testing.expectEqual(@as(u32, 894), exp(320));

    // 3. e^2.75 검증 (정수=2, HighM=12, LowM=0)
    // 수학적 실제값: e^2.75 ≈ 15.64263 -> 고정소수점 변환: 15.64263 * 256 ≈ 4005
    // 입력값: 2.75 * 256 = 704 (바이너리: 00000010 1100 0000)
    try std.testing.expectEqual(@as(u32, 4005), exp(704));

    // 4. 세 테이블이 모두 깨어나는 복합 케이스: e^1.34375 (정수=1, HighM=5, LowM=8)
    // 수학적 실제값: e^1.34375 ≈ 3.83337 -> 고정소수점 변환: 3.83337 * 256 ≈ 981
    // 입력값: 1.34375 * 256 = 344 (바이너리: 00000001 0101 1000)
    try std.testing.expectEqual(@as(u32, 981), exp(344));
}

test "fixed minusexp" {
    // 1. e^-0.25 검증 (입력: -0.25 * 256 = -64)
    // 수학적 실제값: e^-0.25 ≈ 0.7788 -> 고정소수점 변환: 0.7788 * 256 ≈ 199
    try std.testing.expectEqual(@as(u32, 199), exp(-64));

    // 2. e^-1.0 검증 (입력: -1.0 * 256 = -256)
    // 수학적 실제값: e^-1 ≈ 0.3678 -> 고정소수점 변환: 0.3678 * 256 ≈ 94
    try std.testing.expectEqual(@as(u32, 94), exp(-256));

    // 3. e^-3.5 검증 (입력: -3.5 * 256 = -896)
    // 수학적 실제값: e^-3.5 ≈ 0.03019 -> 고정소수점 변환: 0.03019 * 256 ≈ 8
    try std.testing.expectEqual(@as(u32, 8), exp(-896));

    // 4. e^-5.5 검증 (표현 가능한 거의 최하한선, 입력: -5.5 * 256 = -1408)
    // 수학적 실제값: e^-5.5 ≈ 0.00408 -> 고정소수점 변환: 0.00408 * 256 ≈ 1
    try std.testing.expectEqual(@as(u32, 1), exp(-1408));

    try std.testing.expectEqual(@as(u32, 1), exp(-1536)); // -6.0

    // 5. 완벽한 언더플로우 한계점 체크 (e^-6.0 이하)
    // 8비트 정밀도 상 0이 되어야 함
    try std.testing.expectEqual(@as(u32, 0), exp(-5000));
}
