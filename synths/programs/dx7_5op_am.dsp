import("stdfaust.lib");
// 5-op serial FM chain + AM tremolo. Carrier fixed at f0 (identifiable). 12 params.
// op5 -> op4 -> op3 -> op2 -> op1 (carrier), then AM envelope.
f0       = hslider("f0",{f0},50,1000,0.1);
r2       = hslider("r2",{r2},0.25,8,0.01);
r3       = hslider("r3",{r3},0.25,8,0.01);
r4       = hslider("r4",{r4},0.25,8,0.01);
r5       = hslider("r5",{r5},0.25,8,0.01);
l1       = hslider("l1",{l1},0,8,0.01);
l2       = hslider("l2",{l2},0,8,0.01);
l3       = hslider("l3",{l3},0,8,0.01);
l4       = hslider("l4",{l4},0,8,0.01);
l5       = hslider("l5",{l5},0,8,0.01);
am_rate  = hslider("am_rate",{am_rate},0.5,20,0.01);
am_depth = hslider("am_depth",{am_depth},0,1,0.01);
phasor(f) = +(f/ma.SR) ~ ma.frac;
op(f,pm)  = phasor(f)*2*ma.PI + pm : sin;
am_env    = 1 - am_depth + am_depth * sin(phasor(am_rate)*2*ma.PI);
process   = op(f0, op(f0*r2, op(f0*r3, op(f0*r4, op(f0*r5,0)*l5)*l4)*l3)*l2)*l1 * 0.3 * am_env;
