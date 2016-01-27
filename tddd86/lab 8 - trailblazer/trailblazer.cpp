/*marso329 joher316
 * if could not get priorityqueue to work without changing line 332 in pqueue.h
 * from:   return heap.get(0).value;
 * to:     return const_cast<ValueType&>(heap.get(0).value);

*/
#include "costs.h"
#include "pqueue.h"
#include "trailblazer.h"
#include <stack>
#include <queue>
#include <algorithm>
// TODO: include any other headers you need; remove this comment
using namespace std;

vector<Node *> depthFirstSearch(BasicGraph& graph, Vertex* start, Vertex* end) {
    graph.resetData();
    //special case if the user is not that bright
    if (start==end){
        vector<Vertex*> path;
        path.push_back(start);
        start->setColor(GREEN);
        return path;
    }
    //check if we are done
    bool quit=false;
    //a stack to store vertex in
    stack<Vertex*> vertexStack;
    //current node we are checking out
    Vertex* currentNode;
    //keep tracks of which have been discoverad and not
    map<Vertex*,bool> discovered;
    for (Node *node :graph.getNodeSet()){
        discovered[node]=false;
    }

    //keep track of all nodes parents
    map<Vertex*,Vertex*> parent;
    //to get started
    vertexStack.push(start);
    //while we still have stuff in stack and not done
    while ((!vertexStack.empty())&&(!quit)){
        //get a node to checkout
        currentNode=vertexStack.top();
        //set it to green
        currentNode->setColor(GREEN);
        //remove from stack
        vertexStack.pop();
        //if current node is not discovered then we check it out
        if (!discovered[currentNode]){
            //set it to discovered
            discovered[currentNode]=true;
            bool deadEnd=true;
            //check all the currentnodes neighbors
            for (Node *node :graph.getNeighbors(currentNode)){
                //if it is not already discovered we set the nodes parent to current node
                if (!discovered[node]){
                    deadEnd=false;
                    //push it for later exploring
                    vertexStack.push(node);
                    parent[node]=currentNode;
                    node->setColor(YELLOW);
                }
                //if the node is the end node we are done
                if (node==end){
                    quit=true;

                     node->setColor(GREEN);
                     break;
                }

            }
            //if the node has no neighbors to discover it is a dead end
            if (deadEnd){
                currentNode->setColor(GRAY);
            }
        }

    }
    //empty the vertexStack
    while (!vertexStack.empty()){
        vertexStack.pop();
    }
    map<Vertex*,Vertex*>::iterator it=parent.find(end);
    //check if the end node has a parent, that means that we found it
    if (it!=parent.end()){
        currentNode=end;
        //empty the path into the stack since it is backward
        while (currentNode!=start){
            vertexStack.push(currentNode);
            currentNode=parent[currentNode];

        }
        //push the start node
        vertexStack.push(start);
        vector<Vertex*> path;
        //empty the stack into path
        while (!vertexStack.empty()){
            path.push_back(vertexStack.top());
            vertexStack.pop();
        }
        //return path
        return path;
    }
    else{
        //if we did not find end
        vector<Vertex*> path;
        return path;
    }
}

vector<Node *> breadthFirstSearch(BasicGraph& graph, Vertex* start, Vertex* end) {
    graph.resetData();
    //todo probably cut of paths that we have already been on
    //a queue of all paths
    queue<vector<Vertex*>> vertexPathQueue;
    //create a path
    vector<Vertex*> path;
    //push start to the path
    path.push_back(start);
    //push the path to the queue of paths
    vertexPathQueue.push(path);

    Vertex* currentNode;
    vector<Vertex*>  currentPath;
    map<Vertex*,bool> discovered;
    for (Node *node :graph.getNodeSet()){
        discovered[node]=false;
    }
    while (!vertexPathQueue.empty()){
        //get the path from the front of the queue
        currentPath=vertexPathQueue.front();
        vertexPathQueue.pop();
        //get the node from the end of the path
        currentNode=currentPath[currentPath.size()-1];
        currentNode->setColor(GREEN);
        discovered[currentNode]=true;
        //if the last node in this path is the end then we have found our path
        if (currentNode==end){
            currentNode->setColor(GREEN);
            return currentPath;
        }
        //we go through all the current nodes n children and create n new paths.
        for (Node *node :graph.getNeighbors(currentNode)){
            if(!discovered[node]){
                vector<Vertex*> newPath=currentPath;
                newPath.push_back(node);
                vertexPathQueue.push(newPath);
                node->setColor(YELLOW);
                discovered[node]=true;
            }
        }
    }


    path.clear();
    return path;
}

vector<Node *> dijkstrasAlgorithm(BasicGraph& graph, Vertex* start, Vertex* end) {
    graph.resetData();
    //set as predescessor if that is undefined
    Vertex* unDefined;

    //the current node we are checking
    Vertex*  currentNode;

    //sets startnode cost to zero
    start->cost=0.0;

    //create prioqueue
    PriorityQueue<Vertex*> vertexPrioQueue;

    //used to keep track of all predeccesors
    map<Vertex*,Vertex*> predecessor;

    //set all costs, sets predecessor and adds the to the queue
    for (Node *node :graph.getNodeSet()){
        //all nodes but start should have infinity cost
        if (node!=start){
            node->cost=INFINITY;
            predecessor[node]=unDefined;
        }
        //add all nodes to queue
        vertexPrioQueue.enqueue(node,node->cost);

    }
    //keep track of the alternative cost
    double alt;
    //while the queue is not empty
    while (!vertexPrioQueue.isEmpty()){
        //put current node to the one with highest priority
        currentNode= vertexPrioQueue.front();
        vertexPrioQueue.dequeue();
        currentNode->setColor(YELLOW);
        currentNode->visited=true;
        if (currentNode==end){
            break;
        }

        //check all the node's neighbors
        for(Node *node :graph.getNeighbors(currentNode)){
            //if we have not visited that node
            if (!node->visited){
                //we check the alternative cost
                alt=currentNode->cost+graph.getArc(currentNode,node)->cost;
                //if the alternative cost is lower then we set that to our new cost
                if (alt<node->cost){
                    node->cost=alt;
                    predecessor[node]=currentNode;
                    vertexPrioQueue.changePriority(node,alt);

                }
            }
        }
        currentNode->setColor(GREEN);

    }
    //if we havent found end
    if(predecessor[end]==unDefined){
        vector<Vertex*> path;
        return path;
    }
    else{
        //if we have found end we trace through the predecessor map to find the path
        stack<Vertex*> vertexStack;
        vector<Vertex*> path;
        currentNode=end;
        while (currentNode!=start){
            vertexStack.push(currentNode);
            currentNode=predecessor[currentNode];

        }
        vertexStack.push(start);
        while (!vertexStack.empty()){
            path.push_back(vertexStack.top());
            vertexStack.pop();
        }
        return path;
    }
}
//this is the same as dijkstrasAlgorithm except for one thing which is commented
vector<Node *> aStar(BasicGraph& graph, Vertex* start, Vertex* end) {
    graph.resetData();
    Vertex* unDefined;
    Vertex*  currentNode;
    start->cost=0.0;
    PriorityQueue<Vertex*> vertexPrioQueue;
    map<Vertex*,Vertex*> predecessor;
    for (Node *node :graph.getNodeSet()){
        if (node!=start){
            node->cost=INFINITY;
            predecessor[node]=unDefined;
        }
        vertexPrioQueue.enqueue(node,node->cost);;

    }
    double alt;
    while (!vertexPrioQueue.isEmpty()){
        currentNode= vertexPrioQueue.front();

        vertexPrioQueue.dequeue();
        currentNode->setColor(YELLOW);
        currentNode->visited=true;
        if (currentNode==end){
            break;
        }
        for(Node *node :graph.getNeighbors(currentNode)){
            if (!node->visited){
                alt=currentNode->cost+graph.getArc(currentNode,node)->cost;
                if (alt<node->cost){
                    node->cost=alt;
                    predecessor[node]=currentNode;
                    //this is the change, the queuepriority comes from the node cost + the heuristic
                    vertexPrioQueue.changePriority(node,node->cost+node->heuristic((end)));

                }
            }
        }
        currentNode->setColor(GREEN);

    }

    if(predecessor[end]==unDefined){
        vector<Vertex*> path;
        return path;
    }
    else{
        stack<Vertex*> vertexStack;
        vector<Vertex*> path;
        currentNode=end;
        while (currentNode!=start){
            vertexStack.push(currentNode);
            currentNode=predecessor[currentNode];

        }
        vertexStack.push(start);
        while (!vertexStack.empty()){
            path.push_back(vertexStack.top());
            vertexStack.pop();
        }
        return path;
    }
}
