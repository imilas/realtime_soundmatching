import("stdfaust.lib");
amp = hslider("amp",{amp},0.1,1,0.01);
carrier = hslider("carrier",{carrier},1,20,0.01);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
process = no.noise*sineOsc(carrier)*amp;
