# coding: utf-8

def create_solver(proto_path, train_net, snapshot_prefix, snapshot=20000, power=0.9, base_lr=1e-3, momentum=0.5, weight_decay=0.0005):
    with open('/home/wuhuikai/Segmentation/Deeplab_v2/exper/solver_template') as f:
        proto = f.read()
    proto = proto.format(train_net, power, base_lr, momentum, weight_decay, snapshot, snapshot_prefix)
    with open(proto_path, 'w') as f:
        f.write(proto)