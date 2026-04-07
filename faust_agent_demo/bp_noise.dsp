import("stdfaust.lib");

lp_cut = hslider("lp_cut[unit:Hz]", 1800, 100, 8000, 1) : si.smoo;
hp_cut = hslider("hp_cut[unit:Hz]",  200,  20, 3000, 1) : si.smoo;
gain   = hslider("gain",             0.15, 0.0, 0.5, 0.001) : si.smoo;

mono = no.noise : fi.lowpass(3, lp_cut) : fi.highpass(3, hp_cut) * gain;
process = mono, mono;
