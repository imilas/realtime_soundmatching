import("stdfaust.lib");
lp_cut = hslider("lp_cut",{lp_cut},50,1000,1);
hp_cut = hslider("hp_cut",{hp_cut},1,120,1);
process = no.noise:fi.lowpass(3,lp_cut):fi.highpass(10,hp_cut);
