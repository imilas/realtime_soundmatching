import("stdfaust.lib");
carrier = hslider("carrier",{carrier},20,300,0.1);
amp = hslider("amp",{amp},0,10,0.1);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
process = sineOsc(amp)*sineOsc(carrier);
