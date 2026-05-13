import("stdfaust.lib");
lp_cut = hslider("lp_cut",{lp_cut},60,6000,1);
hp_cut = hslider("hp_cut",{hp_cut},30,5000,1);
process = no.noise:fi.lowpass(3,lp_cut):fi.highpass(10,hp_cut);
