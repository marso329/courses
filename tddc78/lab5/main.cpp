#include <iostream>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <random>
#include <algorithm>
#include <ctime>

#ifndef sqr
#define sqr(a) ((a)*(a))
#endif

#define RESOLUTION 0.000001

#define STEP_SIZE 1.0
#define NUMBER_OF_STEPS 10

class Particle {
public:
	Particle(double x, double y, double vx, double vy) :
			x(x), y(y), vx(vx), vy(vy) {

	}

	Particle() :
			x(0), y(0), vx(0), vy(0) {
	}
	void print() {
		std::cout << "x: " << x << " y: " << y << " vx: " << vx << "  vy:" << vy
				<< std::endl;
	}
	inline bool operator==(const Particle& rhs) {
		return this->vx == rhs.vx && this->vy == rhs.vy && this->x == rhs.x
				&& this->y == rhs.y;
	}
	/**x and y is position in meters from origo
	 *  vx and vy is velocity in meters/second
	 */
	double x, y, vx, vy;
protected:
private:
};

class Container {
public:
	Container(double height, double width, double start, double end) :
			complete_height(height), complete_width(width), start_x(start), end_x(
					end) {

	}
	void print() {
		for (auto it = particles.begin(); it != particles.end(); it++) {
			(*it).print();
		}
	}
	void generate_random_particles(int number_of_particles) {
		std::uniform_real_distribution<double> height_distribution(RESOLUTION,
				complete_height - RESOLUTION);
		std::uniform_real_distribution<double> width_distribution(
				start_x + RESOLUTION, end_x - RESOLUTION);
		std::uniform_real_distribution<double> speed_distribution(-35.0, 35.0);
		std::default_random_engine re;
		for (int i = 0; i < number_of_particles; i++) {
			Particle temp = Particle(width_distribution(re),
					height_distribution(re), speed_distribution(re),
					speed_distribution(re));
			particles.push_back(temp);
		}
	}
	void collision(Particle part1, Particle part2) {

	}

	void do_time_step() {
		float collision_time=-1.0;
		for (auto it = particles.begin(); it != particles.end(); it++) {
			auto it2 = std::vector<Particle>::iterator(it);
			it2++;
			for (; it2 != particles.end(); it2++) {
				Particle part1 = *it;
				Particle part2 = *it2;
				collision_time = collide(&part1, &part2);
				if (collision_time != -1.0) {
					interact(&part1, &part2, collision_time);
					break;
				}

			}
			if (collision_time == -1.0) {
				feuler(&(*it), STEP_SIZE);
			}
			//check for wall collision
			Particle part1 = *it;
			if (complete_width < part1.x) {
				part1.x = complete_width;
				part1.vx = -part1.vx;
				momentum += 2.0 * fabs(part1.vx);
			} else if (end_x <= part1.x) {
				particles_to_send_right.push_back(part1);

			} else if (part1.x < 0.0) {
				momentum += 2.0 * fabs(part1.vx);
				part1.x = 0.0;
				part1.vx = -part1.vx;
			} else if (part1.x < start_x) {
				particles_to_send_left.push_back(part1);
			}
			if (part1.y > complete_height) {
				momentum += 2.0 * fabs(part1.vy);
				part1.y = complete_height;
				part1.vy = -part1.vy;

			} else if (part1.y < 0.0) {
				momentum += 2.0 * fabs(part1.vy);
				part1.y = 0.0;
				part1.vy = -part1.vy;
			}

		}

	}

	float collide(Particle *p1, Particle *p2) {
		double a, b, c;
		double temp, t1, t2;

		a = sqr(p1->vx-p2->vx) + sqr(p1->vy - p2->vy);
		b = 2
				* ((p1->x - p2->x) * (p1->vx - p2->vx)
						+ (p1->y - p2->y) * (p1->vy - p2->vy));
		c = sqr(p1->x-p2->x) + sqr(p1->y - p2->y) - 4 * 1 * 1;

		if (a != 0.0) {
			temp = sqr(b) - 4 * a * c;
			if (temp >= 0) {
				temp = sqrt(temp);
				t1 = (-b + temp) / (2 * a);
				t2 = (-b - temp) / (2 * a);

				if (t1 > t2) {
					temp = t1;
					t1 = t2;
					t2 = temp;
				}
				if ((t1 >= 0) & (t1 <= 1))
					return t1;
				else if ((t2 >= 0) & (t2 <= 1))
					return t2;
			}
		}
		return -1;
	}
	void interact(Particle *p1, Particle *p2, float t) {
		float c, s, a, b, tao;
		Particle p1temp, p2temp;

		if (t >= 0) {

			/* Move to impact point */
			(void) feuler(p1, t);
			(void) feuler(p2, t);

			/* Rotate the coordinate system around p1*/
			p2temp.x = p2->x - p1->x;
			p2temp.y = p2->y - p1->y;

			/* Givens plane rotation, Golub, van Loan p. 216 */
			a = p2temp.x;
			b = p2temp.y;
			if (p2->y == 0) {
				c = 1;
				s = 0;
			} else {
				if (fabs(b) > fabs(a)) {
					tao = -a / b;
					s = 1 / (sqrt(1 + sqr(tao)));
					c = s * tao;
				} else {
					tao = -b / a;
					c = 1 / (sqrt(1 + sqr(tao)));
					s = c * tao;
				}
			}

			p2temp.x = c * p2temp.x + s * p2temp.y; /* This should be equal to 2r */
			p2temp.y = 0.0;

			p2temp.vx = c * p2->vx + s * p2->vy;
			p2temp.vy = -s * p2->vx + c * p2->vy;
			p1temp.vx = c * p1->vx + s * p1->vy;
			p1temp.vy = -s * p1->vx + c * p1->vy;

			/* Assume the balls has the same mass... */
			p1temp.vx = -p1temp.vx;
			p2temp.vx = -p2temp.vx;

			p1->vx = c * p1temp.vx - s * p1temp.vy;
			p1->vy = s * p1temp.vx + c * p1temp.vy;
			p2->vx = c * p2temp.vx - s * p2temp.vy;
			p2->vy = s * p2temp.vx + c * p2temp.vy;

			/* Move the balls the remaining time. */
			c = 1.0 - t;
			(void) feuler(p1, c);
			(void) feuler(p2, c);
		}

	}
	int feuler(Particle *a, float time_step) {
		a->x = a->x + time_step * a->vx;
		a->y = a->y + time_step * a->vy;
		return 0;
	}

	void send_data() {
		MPI_Request temp_request, temp_request1, temp_request2, temp_request3;
		int left_send = particles_to_send_left.size();
		int right_send = particles_to_send_right.size();
		if (rank != 0) {
			MPI_Isend(&left_send, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
					&temp_request);
		}
		if (rank != size - 1) {
			MPI_Isend(&right_send, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD,
					&temp_request1);
		}
		MPI_Status temp_status;
		int left_to_receive = 0;
		int right_to_receive = 0;
		if (rank != 0) {
			MPI_Recv(&left_to_receive, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
					&temp_status);
		}
		if (rank != size - 1) {
			MPI_Recv(&right_to_receive, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD,
					&temp_status);
		}
		double temp_data_to_send_left[4 * left_send];
		double temp_data_to_send_right[4 * right_send];
		for (int i = 0; i < left_send; i++) {
			temp_data_to_send_left[i * 4] = particles_to_send_left[i].vx;
			temp_data_to_send_left[i * 4 + 1] = particles_to_send_left[i].vy;
			temp_data_to_send_left[i * 4 + 2] = particles_to_send_left[i].x;
			temp_data_to_send_left[i * 4 + 3] = particles_to_send_left[i].y;
			Particle temp = particles_to_send_left[i];
			auto it = std::find(particles.begin(), particles.end(), temp);
			if (it != particles.end()) {
				particles.erase(it);

			}

		}

		for (int i = 0; i < right_send; i++) {
			temp_data_to_send_right[i * 4] = particles_to_send_right[i].vx;
			temp_data_to_send_right[i * 4 + 1] = particles_to_send_right[i].vy;
			temp_data_to_send_right[i * 4 + 2] = particles_to_send_right[i].x;
			temp_data_to_send_right[i * 4 + 3] = particles_to_send_right[i].y;
			auto it = std::find(particles.begin(), particles.end(),
					particles_to_send_right[i]);
			if (it != particles.end()) {
				particles.erase(it);

			}

		}
		if (rank != 0) {
			MPI_Isend(temp_data_to_send_left, left_send * 4, MPI_DOUBLE,
					rank - 1, 0, MPI_COMM_WORLD, &temp_request2);
		}

		if (rank != size - 1) {
			MPI_Isend(temp_data_to_send_right, right_send * 4, MPI_DOUBLE,
					rank + 1, 0, MPI_COMM_WORLD, &temp_request3);
		}
		double temp_data_to_receive_left[4 * left_to_receive];
		double temp_data_to_receive_right[4 * right_to_receive];
		if (rank != 0) {
			MPI_Recv(temp_data_to_receive_left, left_to_receive * 4, MPI_DOUBLE,
					rank - 1, 0, MPI_COMM_WORLD, &temp_status);
		}
		if (rank != size - 1) {
			MPI_Recv(temp_data_to_receive_right, right_to_receive * 4,
					MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &temp_status);
		}
		Particle temp;
		for (int i = 0; i < left_to_receive; i++) {
			temp = Particle(temp_data_to_receive_left[i * 4],
					temp_data_to_receive_left[i * 4 + 1],
					temp_data_to_receive_left[i * 4 + 2],
					temp_data_to_receive_left[i * 4 + 3]);
			particles.push_back(temp);
		}
		for (int i = 0; i < right_to_receive; i++) {
			temp = Particle(temp_data_to_receive_right[i * 4],
					temp_data_to_receive_right[i * 4 + 1],
					temp_data_to_receive_right[i * 4 + 2],
					temp_data_to_receive_right[i * 4 + 3]);
			particles.push_back(temp);
		}
		particles_to_send_left.clear();
		particles_to_send_right.clear();

	}

	void sum_momentum() {
		MPI_Reduce(&momentum, &total_momentum, 1, MPI_DOUBLE, MPI_SUM, 0,
				MPI_COMM_WORLD);
	}

	std::vector<Particle> particles;
	std::vector<Particle> particles_to_send_left;
	std::vector<Particle> particles_to_send_right;
	double complete_height = 0.0;
	double complete_width = 0.0;
	double start_x = 0.0;
	double end_x = 0.0;
	double momentum = 0.0;
	double total_momentum = 0.0;
	int rank = 0;
	int size = 0;
protected:
private:

};

int main(int argc, char **argv) {
	double starttime, endtime;
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " volume(float) particles(int)"
				<< std::endl;
		exit(1);
	}

	double volume = atof(argv[1]);
	int number_of_particles = atoi(argv[2]);

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
	double container_height = sqrt(volume / (double) size);
	double container_width = sqrt(volume * (double) size);
	double start_x = (container_width / (double) size) * (double) rank;
	double end_x = (container_width / (double) size) * (double) (rank + 1.0);
	Container particles = Container(container_height, container_width, start_x,
			end_x);
	particles.generate_random_particles(number_of_particles / size);
	particles.rank = rank;
	particles.size = size;
	starttime = MPI_Wtime();
	for (int i = 0; i < NUMBER_OF_STEPS; i++) {
		particles.do_time_step();
		particles.send_data();
		if (rank == 0) {
			std::cout << i << std::endl;
		}
	}
	particles.sum_momentum();
	endtime = MPI_Wtime();
	double preassure = particles.total_momentum
			/ (container_height * container_width * NUMBER_OF_STEPS);
	if (rank == 0) {
		std::cout << "the pressure is: " << preassure << std::endl;
		std::cout << "that took " << endtime - starttime << " seconds"
				<< std::endl;
	}

	MPI_Finalize();

}
