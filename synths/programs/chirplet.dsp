import("stdfaust.lib");
increase_speed = hslider("increase_speed",{increase_speed},1,20,0.1);
starting_pitch = hslider("starting_pitch",{starting_pitch},30,100,0.1);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
increasing_pitch(rate) = _ ~ +(rate/ma.SR) : exp;
process = sineOsc(increasing_pitch(increase_speed) + starting_pitch);
