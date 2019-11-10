#include "ros/ros.h"
#include "std_msgs/String.h"
#include <string>
/**
 * Este nodo llamado nodo_emisor emite mensajes "mensajeTest_topic"
del tipo beginner_tutorials::mensajeTest
*/
int main(int argc, char **argv) {
 ros::init(argc, argv, "nodo_emisor"); //registra el nombre del nodo
 ros::NodeHandle nodo; //Creamos un objeto nodo
 std_msgs::String msg;
 ROS_INFO("nodo_informacion personal creado y registrado"); //to screen and file
 std::string n="";

 ros::Publisher publicadorMensajes = nodo.advertise<std_msgs::String>("mensajeTest_topic",0);
 //tiempo a dormir en cada iteracción
 ros::Duration seconds_sleep(1);
 //ejecuta constantemente hasta recibir un Ctr+C
 int contador = 0;
 while (ros::ok()){
 std::cout << "JELLO" << '\n';
 std::cin>> n;
 msg.data=n;
 publicadorMensajes.publish(msg);
 }
}
