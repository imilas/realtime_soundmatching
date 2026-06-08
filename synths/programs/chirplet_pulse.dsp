import("stdfaust.lib");
increase_speed = hslider("increase_speed",{increase_speed},2,7,0.1);
pulse_rate = hslider("pulse_rate",{pulse_rate},1,10,0.1);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
sawOsc(f) = +(f/ma.SR) ~ ma.frac;
increasing_pitch(rate) = _ ~ +(rate/ma.SR) : exp;
process = sineOsc(30 + increasing_pitch(increase_speed) + sawOsc(pulse_rate) * 526);
