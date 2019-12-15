/*
Author: Perkz Zheng
Last Date Modified: Dec 01 2019
Description:d evelop a 3D simulation using MPI and OpenGL to demo the show to the game organizers;

*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "iomanip"
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <GL/glut.h>
#include <chrono>
#include <thread>
#include "ECE_Bitmap.h"
#include <limits>

// Global Variables for loading bitmaps
GLuint texture;

struct Image {

	unsigned long sizeX;

	unsigned long sizeY;

	char* data;

};

typedef struct Image Image;

BMP inBitmap;

// std
using namespace std;

// Global Variables for lookat function
float eye_x = 0.0;
float eye_y = -60.0;
float eye_z = 100.0;
float center_x = 0.0;
float center_y = 0.0;
float center_z = 0.0;

// Global Variables for UVAs movement
double t = 0.1;// 100 msec
double hookeK = 0.2;// Hooke's law coefficient
double surfaceK = 5;// surface movement coefficient
double sphere_center[3] = { 0.0,0.0,50.0 };
double radius = 10.0;
double m = 1.;

// Global Variables for color changing while moving
double red = 255.0;
int times = 0;// times of clolor-changing

// Structure of UAVs information, including status, positions, and velocities
struct UAVs {
	// Define structures of UAVs
	int finished;// = 0, mean unfinished, = 1 means finished
	double x;// positions
	double y;
	double z;
	double Vx;// speed
	double Vy;
	double Vz;
};

// All UAVs (16)
UAVs allUAVs[16];
UAVs drawUAVsPos[16];

// Define new MPI datatype for sending and receiving UAVs structures
MPI_Datatype mpiUAVsType;

// Function Declarations
void changeSize(int w, int h);
void displayFootballField();
void drawUAV();
void init();
void timerFunction(int ID);
void renderScene();
void mainOpenGL(int argc, char** argv);
void calcualteUAVsLocation(UAVs* cur_one, int rankID);
double compositeValue(double v1, double v2, double v3);
void UAVsNear(UAVs* uav);
void swapUAVs(UAVs* v1, UAVs* v2);

int main(int argc, char** argv)
{// Main Function for MPI (16)
	int numTasks, rank;

	int rc = MPI_Init(&argc, &argv);

	if (rc != MPI_SUCCESS)
	{// Construct MPI unsuccessfully
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	// Initialize UAVs structures
	allUAVs[0].finished = drawUAVsPos[0].finished = 0;
	allUAVs[0].x = allUAVs[0].y = allUAVs[0].z = allUAVs[0].Vx = allUAVs[0].Vy = allUAVs[0].Vz = 0.0;
	drawUAVsPos[0].x = drawUAVsPos[0].y = drawUAVsPos[0].z = drawUAVsPos[0].Vx = drawUAVsPos[0].Vy = drawUAVsPos[0].Vz = 0.0;
	int curID = 1;
	for (int j = -1; j < 2; ++j)
	{
		for (int i = -2; i < 3; ++i)
		{
			allUAVs[curID].finished = 0;
			allUAVs[curID].x = i * 25.;
			allUAVs[curID].y = j * 26.5;
			allUAVs[curID].z = 0.;
			allUAVs[curID].Vx = 0.;
			allUAVs[curID].Vy = 0.;
			allUAVs[curID].Vz = 0.;
			drawUAVsPos[curID].finished = 0;
			drawUAVsPos[curID].x = i * 25.;
			drawUAVsPos[curID].y = j * 26.5;
			drawUAVsPos[curID].z = 0.;
			drawUAVsPos[curID].Vx = 0.;
			drawUAVsPos[curID].Vy = 0.;
			drawUAVsPos[curID].Vz = 0.;
			curID++;
		}
	}

	int gsize = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &gsize);

	// Define new MPI datatype
	const int nitems = 7;
	const int lengths[7] = {1,1,1,1,1,1,1 };
	MPI_Datatype types[7] = { MPI_INT,MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE };
	MPI_Aint offsets[7];

	offsets[0] = offsetof(UAVs, finished);
	offsets[1] = offsetof(UAVs, x);
	offsets[2] = offsetof(UAVs, y);
	offsets[3] = offsetof(UAVs, z);
	offsets[4] = offsetof(UAVs, Vx);
	offsets[5] = offsetof(UAVs, Vy);
	offsets[6] = offsetof(UAVs, Vz);

	MPI_Type_create_struct(nitems, lengths, offsets, types, &mpiUAVsType);
	MPI_Type_commit(&mpiUAVsType);

	// Start MPI
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0)
	{
		mainOpenGL(argc, argv);// Master Thread aims at constructing 3D graph
	}
	else
	{
		// Sleep for 5 seconds
		std::this_thread::sleep_for(std::chrono::seconds(5));
		// Every UAV start moving, sending information and receive ones.
		UAVs cur_UAVs = allUAVs[rank];
		for (int ii = 0; ii < 900; ii++)
		{
		// The professor said that I could set it to more than 600 because dispaly function may be quicker than expected
			calcualteUAVsLocation(&cur_UAVs, rank);
			MPI_Allgather(&cur_UAVs, 1, mpiUAVsType, allUAVs, 1, mpiUAVsType, MPI_COMM_WORLD);
		}
		// if it is out of time, finish OpenGL in mainOpenGL();
		cur_UAVs.finished = 1;
		MPI_Allgather(&cur_UAVs, 1, mpiUAVsType, allUAVs, 1, mpiUAVsType, MPI_COMM_WORLD);
	}
	// Finalize MPI;
	MPI_Finalize();

	return 0;
}

void changeSize(int w, int h)
{// Change Windows Size
	float ratio = ((float)w) / ((float)h); // window aspect ratio
	glMatrixMode(GL_PROJECTION); // projection matrix is active
	glLoadIdentity(); // reset the projection
	gluPerspective(80.0, ratio, 0.1, 1000.0); // perspective transformation
	glMatrixMode(GL_MODELVIEW); // return to modelview mode
	glViewport(0, 0, w, h); // set viewport (drawing area) to entire window
}

void displayFootballField()
{// display football field by loading bitmaps
	glPushMatrix();
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	// glDisable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(1, 1);
	double scale = 1.2;
	glVertex3f(50.0*scale, 26.5*scale, 0.0);
	glTexCoord2f(0, 1);
	glVertex3f(50.0*scale, -26.5*scale, 0.0);
	glTexCoord2f(0, 0);
	glVertex3f(-50.0*scale, -26.5*scale, 0.0);
	glTexCoord2f(1, 0);
	glVertex3f(-50.0*scale, 26.5*scale, 0.0);
	glEnd();
	glPopMatrix();
	glDisable(GL_TEXTURE_2D);
}

void drawUAV()
{// Draw One UAV
	glEnable(GL_NORMALIZE);
	glPushMatrix();
	glTranslatef(0.0, 0.0, 1.0);
	glScalef(1.25, 1.25, 1.25);
	glutSolidOctahedron();
	glPopMatrix();
	glEnable(GL_COLOR_MATERIAL);
}

void init()
{// Initialize Lighting and Bitmaps
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.5, 0.5, 0.5, 0.0);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);

	GLfloat light0_ambient[] = { 0.2, 0.2, 0.2, 0.2 };
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);

	// loading bitmaps
	inBitmap.read("field.bmp");
	// construct one texture based on bitmaps
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture
	glTexImage2D(GL_TEXTURE_2D, 0, 3, inBitmap.bmp_info_header.width, inBitmap.bmp_info_header.height, 0,
		GL_BGR_EXT, GL_UNSIGNED_BYTE, &inBitmap.data[0]);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
}

void renderScene()
{// Render 3D Scence

	// Clear color and depth buffers
	glClearColor(0.0, 0.0, 0.0, 1.0); // background color to black
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();

	gluLookAt(eye_x, eye_y, eye_z,
		center_x, center_y, center_z,
		0.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);

	displayFootballField();

	// draw a virtual sphere
	glPushMatrix();
	glColor3f(1.0, 1.0, 1.0);
	glTranslatef(0.0, 0.0, 50.0);
	glutWireSphere(10.0, 15, 15);
	glPopMatrix();

	// The magnitude of the color will oscillate between full and half color
	// The professor in piazza said we could increse the frequency of clolor-changing in order to be more clear
	times++;// the times of changing colors
	if (times % 2 == 0)
	{// change 2 rgb value every 2 second
		int num = times / 2;
		if ((num*2/128)%2 == 0)
		{
			red -= 2;
		}
		else
		{
			red += 2;
		}
	}

	// Draw UAVs
	for (int i = 1; i < 16; ++i)
	{
		auto uav = drawUAVsPos[i];
		glPushMatrix();
		glColor3f(red/255., 0.0, 0.0);
		glTranslatef(uav.x, uav.y, uav.z);
		drawUAV();
		glPopMatrix();
	}

	glutSwapBuffers(); // Make it all visible

	// receive information of all 15 UAVs
	UAVs cur_UAVs = drawUAVsPos[0];
	MPI_Allgather(&cur_UAVs, 1, mpiUAVsType, drawUAVsPos, 1, mpiUAVsType, MPI_COMM_WORLD);
	if (drawUAVsPos[1].finished == 1)
	{// if it is out of time, exit OpenGL
		exit(0);
	}
	// check whether two UAVs is close than 1 m, if it is, swap their speed;
	UAVsNear(drawUAVsPos);
}

void timerFunction(int ID)
{// Draw 3D scene every 100 ms
	glutPostRedisplay();
    glutTimerFunc(80, timerFunction, 0);
    // The professor said that I could decrese interval-time for showing up 3D models, because MPI may be quicker than dispalying.

}

void mainOpenGL(int argc, char** argv)
{// Main function of OpenGL
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(400, 400);
	glutCreateWindow("Final Project By Peng Zheng");	

	init();

	glutReshapeFunc(changeSize);
	glutDisplayFunc(renderScene);
	glutTimerFunc(80,timerFunction,0);
	glutMainLoop();

	return;
}

void calcualteUAVsLocation(UAVs* cur_one, int rankID)
{// calculate one UAV's location after given specific force
	double Fx, Fy, Fz, ax, ay, az;
	double k;
	// calculate the distance and vector to the surface of the sphere
	double dist = pow(pow(cur_one->x - sphere_center[0], 2) + pow(cur_one->y - sphere_center[1], 2) + pow(cur_one->z - sphere_center[2], 2), 0.5);
	double surfacePoint[3];
	surfacePoint[0] = (cur_one->x - sphere_center[0]) * radius / dist + sphere_center[0];
	surfacePoint[1] = (cur_one->y - sphere_center[1]) * radius / dist + sphere_center[1];
	surfacePoint[2] = (cur_one->z - sphere_center[2]) * radius / dist + sphere_center[2];
	double Vx = (surfacePoint[0] - (cur_one->x)) * hookeK;
	double Vy = (surfacePoint[1] - (cur_one->y)) * hookeK;
	double Vz = (surfacePoint[2] - (cur_one->z)) * hookeK;
	double interval_time;
	interval_time = t;
	// Give specific force based on the UAV's vector to the surface (applying Hooke coefficient)
	Fx = m * (Vx) / interval_time;
	Fy = m * (Vy) / interval_time;
	Fz = m * (Vz) / interval_time;

	if (fabs(dist - radius) / radius < 0.02)
	{// If UAV is close to the surface, give it a force pointing to centers to make it move in a circle
		double vectorX = (sphere_center[0] - (cur_one->x));
		double vectorY = (sphere_center[1] - (cur_one->y));
		double vectorZ = (sphere_center[2] - (cur_one->z));
		Fx = m * vectorX;
		Fy = m * vectorY;
		Fz = m * vectorZ;

		double vectorV = pow(pow(cur_one->Vx, 2) + pow(cur_one->Vy, 2) + pow(cur_one->Vz, 2), 0.5);
		if (vectorV < 2.0)
		{// if Velocity is smaller than 2.0, increase Force
			if (rankID == 8)
			{ // First increase the force that starting from the center of football fields
				Fx *= 4.0;
				Fy *= 4.0;
				Fz = 0;
			}
		}
	}

	if (rankID == 8 && fabs(dist - radius) / radius > 0.1 && cur_one->x == 0)
	{// Give UAV starting from the center of football field a force in x direction in order to make it move in a circle
		// instead of staying fluctuated
		Fx = (10.0 - cur_one->Vx) * m * surfaceK / interval_time;
	}

	// if the total Force is larger than 20.0, decrease it by applying an coefficient;
	double F = pow(pow(Fx, 2) + pow(Fy, 2) + pow(Fz, 2), 0.5);
	k = F / 20.0;
	if (k > 1.0)
	{
		Fx /= k;
		Fy /= k;
		Fz = Fz / k - 10;
	}

	// Calculate velocities in all directions
	cur_one->Vx += Fx * interval_time / m;
	cur_one->Vy += Fy * interval_time / m;
	cur_one->Vz += Fz * interval_time / m;
	double V = compositeValue(cur_one->Vx, cur_one->Vy, cur_one->Vz);

	// if velocties is larger than 2.0 and it is far away from the sphere, decrease the force
	if (V / 2.0 > 1.0 && fabs(dist - radius) / radius >= 0.5)
	{
		cur_one->Vx /= V / 2.0;
		cur_one->Vy /= V / 2.0;
		cur_one->Vz /= V / 2.0;
		Fx /= V / 2.0;
		Fy /= V / 2.0;
		Fz /= V / 2.0;
	}

	// Calculate positions based on force and velocities
	cur_one->x += (cur_one->Vx) * interval_time + Fx * interval_time * interval_time / (2 * m);
	cur_one->y += (cur_one->Vy) * interval_time + Fy * interval_time * interval_time / (2 * m);
	cur_one->z += (cur_one->Vz) * interval_time + Fz * interval_time * interval_time / (2 * m);
	return;
}

double compositeValue(double v1, double v2, double v3)
{// calculate composite value based on three directions
	return pow(pow(v1, 2) + pow(v2, 2) + pow(v3, 2), 0.5)/3.2;
}

void UAVsNear(UAVs* uav)
{// when two UAVs are too near, swap their velocities
	for (int i = 1; i < 15; ++i)
	{
		for (int j = i + 1; j < 16; ++j)
		{
			double distanceIJ = pow(pow(uav[i].x - uav[j].x, 2) + pow(uav[i].y - uav[j].y, 2) + pow(uav[i].z - uav[j].z, 2), 0.5);
			if (distanceIJ < 1.0)
			{// less than 1 m
				swapUAVs(&uav[i], &uav[j]);
			}
		}
	}
}

void swapUAVs(UAVs* v1, UAVs* v2)
{// swap two UAVs' velocties in all three directions
	double V[3] = { v1->Vx,v1->Vy,v1->Vz };
	v1 -> Vx = v2 -> Vx;
	v1 -> Vy = v2 -> Vy;
	v1 -> Vz = v2 -> Vz;
	v2 -> Vx = V[0];
	v2 -> Vy = V[1];
	v2 -> Vz = V[2];
	return;
}
