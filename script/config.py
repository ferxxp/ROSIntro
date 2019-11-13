from interaccion.msg import usuario
import rospy
sdmg=usuario()
bi=False
bp=False
be=False
sent_first_message=False
pub=rospy.Publisher('user_topic', usuario, queue_size=10)
