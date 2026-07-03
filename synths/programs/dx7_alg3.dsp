import("stdfaust.lib");
// DX7-style FM, algorithm 3: three modulators (op2,op3,op4) -> one carrier (op1).
// Carrier fixed at f0 to resolve f0/ratio scale ambiguity. 8 params.
f0 = hslider("f0",{f0},50,1000,0.1);
r2 = hslider("r2",{r2},0.25,8,0.01);
r3 = hslider("r3",{r3},0.25,8,0.01);
r4 = hslider("r4",{r4},0.25,8,0.01);
l1 = hslider("l1",{l1},0,8,0.01);
l2 = hslider("l2",{l2},0,8,0.01);
l3 = hslider("l3",{l3},0,8,0.01);
l4 = hslider("l4",{l4},0,8,0.01);
phasor(f) = +(f/ma.SR) ~ ma.frac;
op(f,pm)  = phasor(f)*2*ma.PI + pm : sin;
process   = op(f0, op(f0*r2,0)*l2 + op(f0*r3,0)*l3 + op(f0*r4,0)*l4)*l1 * 0.3;
