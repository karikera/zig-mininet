pub const expIntOffset = 6;
pub const expIntShift = 8;
pub const expIntTable: [11]u32 = .{
    103278,
    280738,
    763125,
    2074389,
    5638775,
    15327780,
    41665227,
    113257828,
    307866697,
    836868447,
    2274844293,
};
pub const expMinusIntOffset = 5;
pub const expMinusIntShift = 24;
pub const expMinusIntTable: [14]u32 = .{
    2489959628,
    916004956,
    336979391,
    123967790,
    45605201,
    16777216,
    6171993,
    2270549,
    835288,
    307285,
    113044,
    41587,
    15299,
    5628,
};
pub const expHighMShift = 16;
pub const expHighMTable: [16]u32 = .{
    65536,
    69763,
    74262,
    79052,
    84150,
    89577,
    95354,
    101504,
    108051,
    115019,
    122437,
    130334,
    138740,
    147688,
    157213,
    167352,
};
pub const expLowMShift = 15;
pub const expLowMTable: [16]u32 = .{
    32768,
    32896,
    33025,
    33154,
    33284,
    33414,
    33545,
    33676,
    33808,
    33940,
    34073,
    34207,
    34341,
    34475,
    34610,
    34745,
};
