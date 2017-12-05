from model import *
from data_utils import SegmentationData
from solver import Solver
import argparse

if __name__ == "__main__":
	# receive arguments from keyboard
	parser.add_argument('-e', type=int, default=10)
	parser.add_argument('-pm', type=str, default="vgg11")
	args = vars(parser.parse_args())

	# initialize the parameters
	epoch = args['e']
	pre_model = args['pm']

	# load the data
	train_data = SegmentationData(image_paths_file='datasets/segmentation_data/train.txt')
	print("Train size: %i" % len(train_data))
	print("Validation size: %i" % len(val_data))
	print("Img size: ", train_data[0][0].size())
	print("Segmentation size: ", train_data[0][1].size())
	print()

	# data loader
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)

	# model
	model = SegmentationNN(pre_model).cuda()

	# train the model
	solver = Solver(
		optimizer=torch.optim.SGD(model.parameters(), lr=5e-5, momentum=0.9, dampening=2e-5),
		loss_func=torch.nn.CrossEntropyLoss(ignore_index=-1)
		)
	model = solver.train(model, train_loader, epoch)

	# save the model
	model.save("models/segmentation_nn.model")

