/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   MPI PageRank							     */
/*   B534 Project1						             */
/*   Group:9                                                                  */
/*   Authors: Jiang Chen, Ninad						     */
/*   Emails: jiang28@indiana.edu      						             */
/*   Date: 							     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>             /* strtok() */
#include <sys/types.h>          /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>             /* getopt() */
#include <mpi.h>
#include <sys/timeb.h>
#include <time.h>
#include "pagerank.h"

int main(int argc, char **argv)
{

  // struct timeval tBegin, tEnd,tDiff;
    
    /* Definition of data structure and variables for MPI PageRank */ 
    int num_urls = 0;
    int totalNumUrls = 0; 
    int **am_index; /* int[num_urls][2] */ 
                    /* am_index[i][0]refers to second index for am,  am_index[i][1] refers to length of target urls list */ 
    int *adjacency_matrix;
    double *rank_values_table;
    double time1;
    double time2;
    double io_time=0;
    double compute_time=0;
    char *input_file; 
    char *output_file;
    int num_iterations;
    double threshold; 
    // gettimeofday(&tBegin, NULL);
    // timeval_print(&tBegin);
    int rank, nproc, mpi_namelen; 
    char mpi_name[MPI_MAX_PROCESSOR_NAME];
 
    /* MPI Initialization */ 
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); 
    MPI_Get_processor_name(mpi_name, &mpi_namelen); 
    
    //Parse command line arguments  
    // When running the program, the command should look like: $ mpirun -np 2 ./mpi_main input_file_path output_file_path iteration threshold
    input_file = argv[1];
    output_file = argv[2];
    num_iterations = atoi(argv[3]);
    threshold = atof(argv[4]);

    if(rank==0){
      time1=MPI_Wtime();
    }

    // Read local adjacency matrix from file for each process  
    mpi_read(input_file, &num_urls, &am_index, &adjacency_matrix, MPI_COMM_WORLD); 
    
    // Set totalNumUrls  
    MPI_Allreduce(&num_urls, &totalNumUrls, 1, MPI_INT, MPI_SUM,MPI_COMM_WORLD); 
 
    // Global page rank value table  
    rank_values_table = (double *) malloc(totalNumUrls * sizeof(double)); 
    assert(rank_values_table != NULL); 
 
     // Broadcast the initial rank values to all other compute nodes  
     MPI_Bcast(rank_values_table, totalNumUrls, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
     if(rank == 0){
	 time2 = MPI_Wtime();
	 io_time = time2-time1;
	 // printf("io time: %f\n", io_time);
       }

     // Start the core computation of MPI PageRank 
     mpi_pagerank(adjacency_matrix, am_index, num_urls, totalNumUrls,  
                  num_iterations, threshold, rank_values_table,  
              MPI_COMM_WORLD);  
     if(rank == 0)
       {
	 time1 = time2;
	 time2 = MPI_Wtime();
	 compute_time= (time2-time1);
	 // printf("compute time: %f\n", compute_time);

       }
    // Save results to a file 
    if(rank == 0) 
     { 
     	sort_output(rank_values_table, totalNumUrls, output_file); 
} 
     
    // Release resources e.g. free(adjacency_matrix);  
    free(adjacency_matrix); 
    free(am_index); 
    free(rank_values_table); 
    
    MPI_Finalize();  
    return (0);  
} 

    int sort_output(double *rank_value_table, int total_num_urls, char *output_file) 
    { 
      int *urls_record = (int *) malloc(total_num_urls * sizeof(int)); 
      int *rank_record =(int *)malloc(total_num_urls * sizeof(int));
      double *values_record = (double *) malloc(total_num_urls * sizeof(double)); 
      int intermediate_url; 
      double intermediate_value; 
      int i; 
      for(i=0;i<total_num_urls;i++) 
	{ 
	  urls_record[i] = i; 
	  values_record[i] = rank_value_table[i];
	  rank_record[i]=i+1;
	} 
	
      //employing bubble sort
      int q; 
      for(q=0;q<total_num_urls;q++) 
	{ 
	  int r; 
	  for(r=0;r<total_num_urls-1;r++) 
	    { 
	      if(values_record[r]<values_record[r+1]) 
		{ 
		  intermediate_value = values_record[r]; 
		  values_record[r] = values_record[r+1]; 
		  values_record[r+1] = intermediate_value; 
		  
		  intermediate_url = urls_record[r]; 
		  urls_record[r] = urls_record[r+1]; 
		  urls_record[r+1] = intermediate_url; 
		} 
	    } 
	} 
      
      /*Write the result to output file*/
      FILE *fout;
      fout = fopen(output_file, "w");
      char cu[10] = "Url";
      char cv[10] = "RankValue";
      char rank[10] ="Rank";
      
      fprintf(fout, "%s",rank);
      fprintf(fout, "\t");
      fprintf(fout, "%s", cu);
      fprintf(fout, "\t");
      fprintf(fout, "%s", cv);
      fprintf(fout, "\n");
      
      int p;
      for(p=0;p<10;p++)
	{
	  fprintf(fout, "%d", rank_record[p]);
	  fprintf(fout, "\t");
	  fprintf(fout, "%d", urls_record[p]);
	  fprintf(fout, "\t");
	  fprintf(fout, "%f", values_record[p]);
	  fprintf(fout, "\n");
	}
      // gettimeofday(&tEnd, NULL);
      // timeval_print(&tEnd);
      // time_t curtime;
      //      int timeval_subtract(struct timeval *result, struct timeval *t2,struct timeval *t1)
      // {
	//	long int diff = (t2->tv_usec+1000000 * t2->tv_sec)-(t1->tv_usec +1000000 *t1->tv_sec);
	//	result->tv_sec =diff/1000000;
	//	result->tv_usec =diff%1000000;
	//	return (diff<0);
	// }
      
      return (0);
    }


