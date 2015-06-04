/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   MPI PageRank							     */
/*   B534 project1 Part2                                                           */
/*   Group:Group9        						             */
/*   Authors: jiang28,							             */
/*   Emails: jiang28@indiana.edu						                     */
/*   Date: 								     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "pagerank.h"
/* Define mpi_pagerank function with following signature */ 
int mpi_pagerank( 
    int *adjacency_matrix, /* adjacency matrix for pagerank */  
    int **am_index, /* index array for adjacency matrix */ 
    int num_urls,        /* num of urls assigned to local machine */ 
    int total_num_urls, /* num of total urls */ 
    int num_iterations, /* num of maximum interations */ 
    double threshold,   /* control the number of iterations */ 
    double *rank_values_table, /* double[total_num_urls] */ 
    MPI_Comm comm)             /* MPI communicator */ 
{
    /* Definitions of variables */ 
    double delta = 0.0;
    double dampingFactor = 0.85;
    int loop = 0;
    
    int source_url, target_url, rank;
    double intermediate_rank_value, danglingValue = 0.0, sumDangling = 0.0;
   	    
    int *target_urls_list;
    double *local_rank_values_table = (double *) malloc(total_num_urls * sizeof(double));
    double *old_rank_values_table = (double *) malloc(total_num_urls * sizeof(double));
  
    /* Get MPI rank */ 
    MPI_Comm_rank(comm, &rank); 
 
    /* Initialized page rank */ 
    int i;
    for(i=0;i<total_num_urls;i++)
    {
    	rank_values_table[i] = 1.0 / total_num_urls;
    }
 
    /* Start computation loop */ 
    do 
    { 
        /* Compute pagerank and dangling values */ 
        int j;
        for(j=0;j<total_num_urls;j++)
        {
        	local_rank_values_table[j] = 0.0;
        	old_rank_values_table[j] = rank_values_table[j];
        }
        danglingValue = 0.0;
        
        int k;
        for(k=0;k<num_urls;k++)
        {
        	int index = am_index[k][0] - am_index[0][0];
        	source_url = adjacency_matrix[index];
        	int num_target_urls = am_index[k][1] - 1;
        	if(num_target_urls != 0)
        	{	
        		target_urls_list = (int *) malloc((num_target_urls) * sizeof(int));
        		int l;
        		for(l=0;l<num_target_urls;l++)
        		{
        			target_urls_list[l] = adjacency_matrix[index+1+l];
        		}
        	}
        	int outdegree_of_source_url = num_target_urls;
        	int m;
        	for(m=0;m<outdegree_of_source_url;m++)
        	{
        		target_url = target_urls_list[m];
        		intermediate_rank_value = local_rank_values_table[target_url] + rank_values_table[source_url] / outdegree_of_source_url;
        		local_rank_values_table[target_url] = intermediate_rank_value;
        	}
        	if(outdegree_of_source_url == 0)
        	{
        		danglingValue += rank_values_table[source_url];
			danglingValue=danglingValue;
        	}
        }
 
        /* Distribute pagerank values */ 
        MPI_Allreduce(local_rank_values_table, rank_values_table, 
                      total_num_urls, MPI_DOUBLE, MPI_SUM, comm); 
 
        /* Distribute dangling values */ 
        MPI_Allreduce(&danglingValue, &sumDangling, 1, MPI_DOUBLE, MPI_SUM, comm); 

 
        /* Root(process 0) computes delta to determine to stop or continue */ 
        if(rank == 0)
        {
		printf("demo here");
        	double dangling_value_per_page = sumDangling / total_num_urls;
        	int o;
        	for(o=0;o<total_num_urls;o++)
        	{
        		rank_values_table[o] = rank_values_table[o] + dangling_value_per_page;
        	}
        	int p;
        	for(p=0;p<total_num_urls;p++)
        	{
        		rank_values_table[p] = dampingFactor * rank_values_table[p] + (1-dampingFactor) * (1.0/total_num_urls);
        	}
        	
        	delta = 0.0;
        	double dif = 0.0;
        	int q;
        	for(q=0;q<total_num_urls;q++)
        	{
        		dif = old_rank_values_table[q] - rank_values_table[q];
			//printf("%f",dif);
			//printf("\n");
        		delta += dif * dif;
        		old_rank_values_table[q] = rank_values_table[q];
        	}
        }
 
        /* Root broadcasts delta */ 
        MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
        MPI_Bcast(rank_values_table, total_num_urls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } 

    while (delta>threshold && loop++ < num_iterations);
    return 1; 
}

