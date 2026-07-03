import("stdfaust.lib");
// DX7-style FM alg1 (serial chain) + AM tremolo + resonant bandpass. 13 params.
// op4 -> op3 -> op2 -> op1 (carrier), then AM envelope, then bandpass filter.
// Fully differentiable (nested sines, IIR bandpass, no feedback/floor/sample-hold).
f0       = hslider("f0",{f0},50,1000,0.1);
r1       = hslider("r1",{r1},0.25,8,0.01);
r2       = hslider("r2",{r2},0.25,8,0.01);
r3       = hslider("r3",{r3},0.25,8,0.01);
r4       = hslider("r4",{r4},0.25,8,0.01);
l1       = hslider("l1",{l1},0,8,0.01);
l2       = hslider("l2",{l2},0,8,0.01);
l3       = hslider("l3",{l3},0,8,0.01);
l4       = hslider("l4",{l4},0,8,0.01);
am_rate  = hslider("am_rate",{am_rate},0.5,20,0.01);
am_depth = hslider("am_depth",{am_depth},0,1,0.01);
bp_fc    = hslider("bp_fc",{bp_fc},200,8000,1);
bp_q     = hslider("bp_q",{bp_q},0.5,10,0.01);
phasor(f) = +(f/ma.SR) ~ ma.frac;
op(f,pm)  = phasor(f)*2*ma.PI + pm : sin;
am_env    = 1 - am_depth + am_depth * sin(phasor(am_rate)*2*ma.PI);
process   = op(f0*r1, op(f0*r2, op(f0*r3, op(f0*r4,0)*l4)*l3)*l2)*l1
            * 0.3 * am_env : fi.resonbp(bp_fc, bp_q);
