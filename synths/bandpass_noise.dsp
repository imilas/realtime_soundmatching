import("stdfaust.lib");

// --- Bandpass noise synth ---
// Two independently controllable cutoffs so the agent
// can be started far from the target and must converge.

hp_freq = hslider("hp_freq[osc:/bandpass_noise/hp_freq]", 20,   20, 8000, 1);
lp_freq = hslider("lp_freq[osc:/bandpass_noise/lp_freq]", 1000, 20, 8000, 1);

process = no.noise * 1 
        : fi.highpass(2, hp_freq)
        : fi.lowpass(2,  lp_freq)
        <: (_, _);   // stereo out
