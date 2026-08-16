#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker

class TableMeshMarker(Node):
    def __init__(self):
        super().__init__("table_mesh_marker")

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL  # RViz 늦게 켜도 보임
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(Marker, "/scene/table_marker", qos)
        self.timer = self.create_timer(0.5, self.publish_once)
        self.published = False

    def publish_once(self):
        if self.published:
            return

        m = Marker()
        m.header.frame_id = "table_link"
        m.ns = "scene"
        m.id = 1
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # package:// 경로로 메시 로드
        m.mesh_resource = "package://franka_kistar_description/meshes/table/soldering_table.obj"
        m.mesh_use_embedded_materials = True

        # 메시 스케일 조정(필요하면)
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # table_link 원점 기준 오프셋이 필요하면 여기서 조절
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0

        # obj에 material 없을 수 있어서 색도 지정(안 보이면 알파부터 확인)
        m.color.a = 1.0
        m.color.r = 0.7
        m.color.g = 0.7
        m.color.b = 0.7

        self.pub.publish(m)
        self.published = True

def main():
    rclpy.init()
    node = TableMeshMarker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
