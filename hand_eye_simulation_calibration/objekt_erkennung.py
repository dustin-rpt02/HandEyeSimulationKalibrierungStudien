import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from sensor_msgs_py import point_cloud2
import open3d as o3d

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

		test = point_cloud2.read_points_list(msg, skip_nans=True)
		print(type(test))
		test = np.array(test, dtype=np.float32)
		test = np.delete(test, 3, 1)
		print(test[0])

		template_ = o3d.geometry.PointCloud()
		template_.points = o3d.utility.Vector3dVector(test)
		o3d.io.write_point_cloud("src/test_point_cloud/data/pointcloud_guitar2.pcd", template_)
		o3d.visualization.draw_geometries([template_])
		
        
def main(args=None):
	rclpy.init(args=args)
	node = Objekt_Erkennung()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
    
if __name__ == "__main__":
    main()