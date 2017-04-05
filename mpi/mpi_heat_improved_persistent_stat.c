#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>
#ifndef NXPROB
#define NXPROB 20
#endif
#ifndef NYPROB
#define NYPROB 20
#endif
#ifndef STEP
#define STEP 30
#endif
#ifndef STEPS
#define STEPS 100
#endif
#ifndef THREAD_COUNT
#define THREAD_COUNT 4
#endif
#define BEGIN       1                  /* message tag */
#define UPDT        2                  /* message tag */
#define END        3                  /* message tag */
#define NONE        0                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

struct Parms {
  float cx;
  float cy;
} parms = {0.1, 0.1};


int main (int argc, char *argv[])
{
void inidat(), prtdat();
int    taskid,                  /* this task's unique id */
    numtasks,                   /* number of tasks */
    dest, source,               /* to - from for message send-receive */
    msgtype,                    /* for message types */
    i,j,it,row,col;             /* loop variables */
MPI_Status status;
int start,end;
double starttime,endtime;
static float  u[2][NXPROB+2][NYPROB+2];        /* array for grid +2 for halo zones*/
/* First, find out my taskid and how many tasks are running */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    int dims[2]={0,0};
    MPI_Dims_create(numtasks,2,dims);
    int dx=dims[0];
    int dy=dims[1];

    int period[2]={0,0};

    //creating cartesian communicator
    MPI_Comm cartcomm;
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,period,0,&cartcomm);

    int mycoords[2];
    MPI_Cart_coords(cartcomm,taskid,2,mycoords);
    int taskx=mycoords[0];
    int tasky=mycoords[1];
    int north,south,east,west;
    //finding neighbors
    MPI_Cart_shift(cartcomm,0,1,&north,&south);
    MPI_Cart_shift(cartcomm,1,1,&east,&west);

    //determine block size
    int myxsize=NXPROB/dx;
    int myysize=NYPROB/dy;
    int xoffset= taskx*myxsize;
    int yoffset= tasky*myysize;

    //contiguous for rows
    MPI_Datatype row_type;
    MPI_Type_contiguous(myysize, MPI_FLOAT, &row_type);
    MPI_Type_commit(&row_type);
    //vector for columns
    MPI_Datatype col_type;
    MPI_Type_vector(myxsize,1,NYPROB+2,MPI_FLOAT, &col_type);
    MPI_Type_commit(&col_type);
    //use of 1D array as a 2D for better performance 2 arrays one for old values one for new values
    if (taskid == MASTER) {
        float  s[2][NXPROB][NYPROB];        /* array for grid*/
        starttime=MPI_Wtime();
       /************************* master code *******************************/
        printf ("Starting mpi_heat2D with %d worker tasks.\n", numtasks);
        /* Initialize grid */
#ifndef CONVERGE
        printf("Grid size: X= %d  Y= %d  Time steps= %d\n",NXPROB,NYPROB,STEPS);
#else
        printf("Grid size: X= %d  Y= %d  Time steps= - \n",NXPROB,NYPROB);
#endif
        inidat(NXPROB, NYPROB, s);
        prtdat(NXPROB, NYPROB, s, "initial_im.dat");
        int coords[2]={0,0};
        for(i=1;i<numtasks;i++){
            MPI_Cart_coords(cartcomm,i,2,coords);
            int sx=coords[0];
            int sy=coords[1];
            int xoffset= sx*myxsize;
            int yoffset= sy*myysize;
            //sending data to workers
            for(row=xoffset;row<xoffset+myxsize;row++){
                MPI_Send(&s[0][row][yoffset],1,row_type,i,BEGIN,MPI_COMM_WORLD);
            }
            //printf("Work sent to task %d (%d,%d)\n",i,sx,sy);
        }
        //master is a worker too so he gets his own data
        for(row=xoffset;row<xoffset+myxsize;row++){
            for(col=yoffset;col<yoffset+myysize;col++){
                u[0][row+1][col+1]=s[0][row][col];
            }
        }
    }
    if (taskid != MASTER)
    {
        //receive data from master
        source = MASTER;
        msgtype = BEGIN;
        for(row=xoffset+1;row<xoffset+1+myxsize;row++){
            MPI_Recv(&u[0][row][yoffset+1],1,row_type,source,msgtype,MPI_COMM_WORLD,&status);
        }
    }
    /*printf("Task %d (%d,%d) %d x %d y started working.\n",taskid,taskx,tasky,myxsize,myysize);*/
    int old=0;
    MPI_Request reqs_send[2][4],reqs_rcv[2][4];
    //persistent sends/recvs one for each array old / 1-old
        /*North send*/
    MPI_Send_init(&u[0][xoffset+1][yoffset+1],1,row_type,north,UPDT,cartcomm,&reqs_send[0][0]);
    MPI_Send_init(&u[1][xoffset+1][yoffset+1],1,row_type,north,UPDT,cartcomm,&reqs_send[1][0]);
        /*South send*/
    MPI_Send_init(&u[0][xoffset+myxsize][yoffset+1],1,row_type,south,UPDT,cartcomm,&reqs_send[0][1]);
    MPI_Send_init(&u[1][xoffset+myxsize][yoffset+1],1,row_type,south,UPDT,cartcomm,&reqs_send[1][1]);
        /*East send*/
    MPI_Send_init(&u[0][xoffset+1][yoffset+1],1,col_type,east,UPDT,cartcomm,&reqs_send[0][2]);
    MPI_Send_init(&u[1][xoffset+1][yoffset+1],1,col_type,east,UPDT,cartcomm,&reqs_send[1][2]);
        /*West send*/
    MPI_Send_init(&u[0][xoffset+1][yoffset+myysize],1,col_type,west,UPDT,cartcomm,&reqs_send[0][3]);
    MPI_Send_init(&u[1][xoffset+1][yoffset+myysize],1,col_type,west,UPDT,cartcomm,&reqs_send[1][3]);
        /*North receive*/
    MPI_Recv_init(&u[0][xoffset][yoffset+1],1,row_type,north,UPDT,cartcomm,&reqs_rcv[0][0]);
    MPI_Recv_init(&u[1][xoffset][yoffset+1],1,row_type,north,UPDT,cartcomm,&reqs_rcv[1][0]);
        /*South receive*/
    MPI_Recv_init(&u[0][xoffset+myxsize+1][yoffset+1],1,row_type,south,UPDT,cartcomm,&reqs_rcv[0][1]);
    MPI_Recv_init(&u[1][xoffset+myxsize+1][yoffset+1],1,row_type,south,UPDT,cartcomm,&reqs_rcv[1][1]);
        /*East receive*/
    MPI_Recv_init(&u[0][xoffset+1][yoffset],1,col_type,east,UPDT,cartcomm,&reqs_rcv[0][2]);
    MPI_Recv_init(&u[1][xoffset+1][yoffset],1,col_type,east,UPDT,cartcomm,&reqs_rcv[1][2]);
        /*West receive*/
    MPI_Recv_init(&u[0][xoffset+1][yoffset+myysize+1],1,col_type,west,UPDT,cartcomm,&reqs_rcv[0][3]);
    MPI_Recv_init(&u[1][xoffset+1][yoffset+myysize+1],1,col_type,west,UPDT,cartcomm,&reqs_rcv[1][3]);
#ifdef CONVERGE
    int valr,val,state,target=STEP-1,conv=0;
#endif
    for (it = 0; it <= STEPS; it++){
        MPI_Startall(4,reqs_send[old]);
        MPI_Startall(4,reqs_rcv[old]);
        //UPDATE INNER INDEPENDENT ELEMENTS
        #ifdef OMPCH
            #pragma omp parallel for num_threads(THREAD_COUNT) private(j)
        #endif
        for(i=xoffset+2;i<xoffset+myxsize;i++){
            for(j=yoffset+2;j<yoffset+myysize;j++){
                u[1-old][i][j]=u[old][i][j]+
                                        parms.cx*(u[old][i+1][j]+
                                        u[old][i-1][j]
                                        -2.0*u[old][i][j])+
                                        parms.cy * (u[old][i][j+1]+
                                        u[old][i][j-1]-
                                        2.0*(u[old][i][j]));
            }
        }
        MPI_Waitall(4, reqs_rcv[old], MPI_STATUS_IGNORE);
        //update blocks
        start=xoffset+1;
        end=xoffset+myxsize+1;
        #ifdef OMPCH
            #pragma omp parallel for num_threads(THREAD_COUNT) private(j)
        #endif
        for(i=start;i<end;i++){
            //east/west
            j=yoffset+1;
            if(i-1 !=0 && i != NXPROB && j-1 !=0 && j !=NYPROB){
                u[1-old][i][j]=u[old][i][j]+
                                        parms.cx*(u[old][i+1][j]+
                                        u[old][i-1][j]
                                        -2.0*u[old][i][j])+
                                        parms.cy * (u[old][i][j+1]+
                                        u[old][i][j-1]-
                                        2.0*(u[old][i][j]));
            }
            j=yoffset+myysize;
            if(i -1 !=0 && i != NXPROB &&j-1 !=0 && j !=NYPROB){
                u[1-old][i][j]=u[old][i][j]+
                                        parms.cx*(u[old][i+1][j]+
                                        u[old][i-1][j]
                                        -2.0*u[old][i][j])+
                                        parms.cy * (u[old][i][j+1]+
                                        u[old][i][j-1]-
                                        2.0*(u[old][i][j]));
            }
        }
        //update north south blocks
        start=yoffset+2;
        end=yoffset+myysize;
        #ifdef OMPCH
            #pragma omp parallel for num_threads(THREAD_COUNT) private(i)
        #endif
        for(j=start;j<end;j++){
            i=xoffset+1;
            if(i -1 !=0 &&  i != NXPROB && j-1 !=0 && j !=NYPROB){
                    u[1-old][i][j]=u[old][i][j]+
                                        parms.cx*(u[old][i+1][j]+
                                        u[old][i-1][j]
                                        -2.0*u[old][i][j])+
                                        parms.cy * (u[old][i][j+1]+
                                        u[old][i][j-1]-
                                        2.0*(u[old][i][j]));
            }
            i=xoffset+myxsize;
            if(i -1 !=0 &&  i != NXPROB &&j-1 !=0 && j !=NYPROB){
                u[1-old][i][j]=u[old][i][j]+
                                        parms.cx*(u[old][i+1][j]+
                                        u[old][i-1][j]
                                        -2.0*u[old][i][j])+
                                        parms.cy * (u[old][i][j+1]+
                                        u[old][i][j-1]-
                                        2.0*(u[old][i][j]));
                        }
        }
#ifdef CONVERGE
        if(it==target){
            valr=0;
            target=it+STEP;
            state=0;
            val=1;
            /*if(taskid==MASTER)*/
              /*printf("%d \n",it);*/
            for(i=xoffset+1;i<xoffset+myxsize+1;i++){
                for(j=yoffset+1;j<myysize+yoffset+1;j++){
                    if (fabs(u[old][i][j]-u[1-old][i][j]) > 1e-3){
                        val= 0;
                        state=1;
                        break;
                    }
                }
                if(state){
                    break;
                }
            }
            MPI_Allreduce(&val,&valr,1,MPI_INT,MPI_LAND,MPI_COMM_WORLD);
            if(valr){
                old=1-old;
		conv=1;
                break;
            }
        }
#endif
        MPI_Waitall(4, reqs_send[old], MPI_STATUS_IGNORE);
        old = 1 - old;
    }
    MPI_Request_free(reqs_send[0]);
    MPI_Request_free(reqs_send[1]);
    MPI_Request_free(reqs_rcv[0]);
    MPI_Request_free(reqs_rcv[1]);
    if (taskid != MASTER)
    {
        dest = MASTER;
        msgtype = END;
        /*printf("%d my rank (%d,%d) my el is %f\n",taskid,taskx,tasky,myar[old][index2D(1,1,myysize+2)]);*/
        for(row=xoffset+1;row<xoffset+myxsize+1;row++){
            MPI_Send(&u[old][row][yoffset+1],1,row_type,dest,msgtype,MPI_COMM_WORLD);
        }
    }
    if (taskid == MASTER){
        float  s[NXPROB][NYPROB];        /* array for grid*/
        int coords[2]={0,0};
        for(row=xoffset;row<xoffset+myxsize;row++){
            for(col=yoffset;col<yoffset+myysize;col++){
                s[row][col]=u[old][row+1][col+1];
            }
        }
        for(i=1;i<numtasks;i++){
            MPI_Cart_coords(cartcomm,i,2,coords);
            int sx=coords[0];
            int sy=coords[1];
            int xoffset= sx*myxsize;
            int yoffset= sy*myysize;
            for(row=xoffset;row<xoffset+myxsize;row++){
                MPI_Recv(&s[row][yoffset],1,row_type,i,END,MPI_COMM_WORLD,&status);
            }
            /*printf("received from task %d\n",i);*/
        }
        endtime=MPI_Wtime();
       prtdat(NXPROB, NYPROB, &s[0][0], "final_im.dat");
#ifdef CONVERGE
	if(conv)
	        printf("Converged after %d steps\n",it);
	else
		printf("Didn't converged\n");
#endif
        printf("Elapsed time %f secs\n",endtime-starttime);
    }
    MPI_Finalize();
    return 0;
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float *u) {
int ix, iy;

for (ix = 0; ix <= nx-1; ix++)
  for (iy = 0; iy <= ny-1; iy++)
     *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *unew, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (iy = ny-1; iy >= 0; iy--) {
  for (ix = 0; ix <= nx-1; ix++) {
    fprintf(fp, "%6.1f", *(unew+ix*ny+iy));
    if (ix != nx-1)
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}
