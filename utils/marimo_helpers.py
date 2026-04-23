import marimo as mo


def query_synth_params(params, osc_host="127.0.0.1", osc_port=5510, timeout=0.5):
    """
    Query a running Faust synth for its current parameter values via OSC.

    Sends '?' to each parameter address and collects responses.
    Returns a dict of {param_name: current_value}, or None for params
    that didn't respond.

    Usage:
        from agent.params import FaustParams
        params = FaustParams("synths/bandpass_noise.dsp.json")
        print(query_synth_params(params))
    """
    import socket
    from pythonosc.osc_message_builder import OscMessageBuilder
    from pythonosc.osc_packet import OscPacket
    import time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))  # ephemeral port
    sock.settimeout(timeout)

    # Send '?' query for each parameter
    for name, p in params.items():
        msg = OscMessageBuilder(address=p.osc_address)
        msg.add_arg("?")
        sock.sendto(msg.build().dgram, (osc_host, osc_port))

    # Collect responses
    results = {name: None for name in params.names()}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, _ = sock.recvfrom(4096)
            for timed_msg in OscPacket(data).messages:
                address = timed_msg.message.address
                args = timed_msg.message.params
                if args:
                    for name, p in params.items():
                        if p.osc_address == address:
                            results[name] = float(args[0])
                            break
        except socket.timeout:
            break

    sock.close()

    print(f"Synth parameters @ {osc_host}:{osc_port}")
    for name, val in results.items():
        p = params[name]
        val_str = f"{val:.4g}" if val is not None else "no response"
        print(f"  {name:20s} = {val_str:>10}  (range [{p.min_val}, {p.max_val}])")

    return results


def make_slider(params, values=None, on_change_func=None):
    """
    on_change_func signature: on_change_func(name, value)
    Each slider gets its own callback that identifies which param changed.
    """
    if values is None:
        values = {}

    def _make_cb(param_name):
        def cb(value):
            if on_change_func:
                on_change_func(param_name, value)
        return cb

    return {
        name: mo.ui.slider(
            start=p.min_val,
            stop=p.max_val,
            value=values.get(name, p.default),
            step=p.step,
            label=name,
            show_value=True,
            on_change=_make_cb(name) if on_change_func else None,
        )
        for name, p in params.items()
    }
