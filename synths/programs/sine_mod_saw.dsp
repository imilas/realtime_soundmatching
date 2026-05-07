import("stdfaust.lib");
carrier = hslider("carrier",{carrier},20,1000,1);
amp = hslider("amp",{amp},1,20,1);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
sawOsc(f) = +(f/ma.SR) ~ ma.frac;
process = sineOsc(amp)*sawOsc(carrier);
