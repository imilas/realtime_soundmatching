import("stdfaust.lib");
amp = hslider("amp",{amp},0,5,0.01);
carrier = hslider("carrier",{carrier},0,4,0.01);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
process = no.noise*sineOsc(carrier)*amp;
