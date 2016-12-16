# coding: utf-8

def create_solver(proto_path, train_net, snapshot_prefix, display=20, snapshot=2000, power=0.9, base_lr=1e-8, momentum=0.9, weight_decay=0.0005, max_iter=20000, iter_size=8):
    with open('solver_template') as f:
        proto = f.read()
    proto = proto.format(train_net, power, base_lr, display, iter_size, max_iter, momentum, weight_decay, snapshot, snapshot_prefix)
    with open(proto_path, 'w') as f:
        f.write(proto)
