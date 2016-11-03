#ifndef  TIME_MEASURING
#define  TIME_MEASURING
#include <sys/time.h>
#include <time.h> 

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
                    (t1->tv_usec + resolution * t1->tv_sec) ;

    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<typename Func>
unsigned long measureFunction(Func f) {
    // Prepare time measuring
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    // Call function to measure
    f();

    // Calculate time spent
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return (t_diff.tv_sec*1e6+t_diff.tv_usec); 
}

template<typename Func>
void measureAndPrint(const char* name, Func f) {
    unsigned long elapsed = measureFunction(f);
    printf("%s runs in %lu µs\n", name, elapsed);
}

#endif
