import subprocess

path = '/home/wuhuikai/Segmentation/Deeplab_v2/exper/'
for i in xrange(1000, 20001, 1000):
	subprocess.call("python {}Test.py 0 deeplab_v2_pairwise_loss finetune_iter_{} fc8".format(path, i), shell=True)
	subprocess.call("python {}PostNoneEval.py 16 deeplab_v2_pairwise_loss fc8 finetune".format(path), shell=True)
