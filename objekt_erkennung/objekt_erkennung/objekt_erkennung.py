import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from sensor_msgs_py import point_cloud2
import open3d as o3d
import torch
import os
import argparse
from learning3d.models import PointNet, Classifier
from learning3d.data_utils import ClassificationData, ModelNet40Data

class Objekt_Erkennung(Node):
	def __init__(self):
		super().__init__('objekt_erkennung')

		self.subscription = self.create_subscription(
			PointCloud2,
			'/points2',
			self.retrieve_camera_data,
			10)
		self.subscription

	def retrieve_camera_data(self, msg):
		self.subscription.destroy()

		point_cloud = point_cloud2.read_points_list(msg, skip_nans=True)
		#print(type(test))
		point_cloud_array = np.array(point_cloud, dtype=np.float32)
		point_cloud_array = np.delete(point_cloud_array, 3, 1)
		#print(test[0])

		"""template_ = o3d.geometry.PointCloud()
		template_.points = o3d.utility.Vector3dVector(test)
		o3d.io.write_point_cloud("src/test_point_cloud/data/pointcloud_guitar2.pcd", template_)
		o3d.visualization.draw_geometries([template_])"""

		self.filter_point_cloud(point_cloud_array)



	def filter_point_cloud(self, point_cloud_array):
		max_z = 0.606
		point_cloud_array = point_cloud_array[(point_cloud_array[:,2] < max_z)]

		self.get_object_position(point_cloud_array)


	def get_object_position(self, point_cloud_array):
		mittelpunkt = np.mean(point_cloud_array, axis=0)
		point_cloud_array -= mittelpunkt
		mittelpunkt = np.mean(point_cloud_array, axis=0)

		self.scale_object(point_cloud_array)



	def scale_object(self, point_cloud_array):
		arr_max = point_cloud_array.max(axis=0)
		arr_min = point_cloud_array.min(axis=0)

		skalierung = 0.9
		multiplicator_matrix = np.concatenate((1/arr_max*skalierung, -1/arr_min*skalierung))

		multiplicator = multiplicator_matrix[np.argmin(multiplicator_matrix)]
		point_cloud_array *= multiplicator

		self.predict_object(point_cloud_array)



	def predict_object(self, point_cloud_array):
		"""for index in range(2000):
		random_index = np.random.randint(0, len(array))
		array = np.delete(array, random_index, 0)"""

		output = self.classify_object(point_cloud_array)[0]
		testset = ClassificationData(ModelNet40Data(train=False))

		values = torch.Tensor([output[6], output[7], output[9], output[17]])
		labels = [testset.get_shape(6), testset.get_shape(7), testset.get_shape(9), testset.get_shape(17)]
		prediction = labels[torch.argmax(values).item()]

		#print(labels)
		#print(values)
		print("Predicted Label:    ", prediction)

		"""template_ = o3d.geometry.PointCloud()
		template_.points = o3d.utility.Vector3dVector(array)
		o3d.visualization.draw_geometries([template_])"""



	def classify_object(self, input):
		args = self.model_options()
	
		my_x = np.array([input])
		points = torch.Tensor(my_x)

		if not torch.cuda.is_available():
			args.device = 'cpu'
		args.device = torch.device(args.device)

		ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
		model = Classifier(feature_model=ptnet)
		if args.pretrained:
			assert os.path.isfile(args.pretrained)
			model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
		model.to(args.device)
		model.eval()

		points = points.to(args.device)
		return model(points)
	


	def model_options(self):
		parser = argparse.ArgumentParser(description='Point Cloud Registration')
		parser.add_argument('--emb_dims', default=1024, type=int,
							metavar='K', help='dim. of the feature vector (default: 1024)')
		parser.add_argument('--pretrained', default='learning3d/pretrained/exp_classifier/models/best_model.t7', type=str,
							metavar='PATH', help='path to pretrained model file (default: null (no-use))')
		parser.add_argument('--device', default='cuda:0', type=str,
							metavar='DEVICE', help='use CUDA if available')

		args = parser.parse_args()
		return args

		
        
def main(args=None):
	rclpy.init(args=args)
	node = Objekt_Erkennung()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
    
if __name__ == "__main__":
    main()