# Multi-dimensional Indexing

This environment runs a column-oriented in-memory database
to compute an aggregate over a range predicate, i.e. SQL queries of the form:

```
SELECT COUNT(*)
FROM Table
WHERE c1 <= C <= c2
  AND d1 <= D <= d2
  AND ...
```
C, D are attributes of Table, and c1, c2, d1, d2 are constants.

The goal of the RL agent is to find the indexing scheme that produces the
lowest average query time for a given dataset
and query workload distribution.

## Layout

At its core, an index is simply a way to lay out data points in storage so that,
given the region defined by the query, the points in that region can be identified
as quickly as possible. An index must produce all points that match a query but should
avoid producing too many non-matches. In this environment, the layout is constrained
to be a grid: if `k` dimensions are indexed, then `k-1` of them are used to form the grid
into which the points are bucketed. The last indexed dimension is used to sort the points
within each bucket.

A layout is fully determined by the following parameters:
1. A choice of `k` dimensions to index (out of all available attributes in the dataset).
1. An ordering of those `k` dimensions. The first `k-1` determine the grid as explained above.
The last one is the sort dimension.
1. For each of the first `k-1` dimensions, the number of columns over that dimension in the grid.

## Dataset

The dataset consists of all entries added to the [Open Street Maps (OSM)](www.openstreetmap.org)
database in the U.S. Northeast. There are 6 attributes, all integers, which are (in order):
* User ID: a unique number identifying the user who added the entry
* Latitude: the point's latitude (GPS coordinates have resolutions of 1e-7, so they are scaled
  accordingly into integers)
* Longtitude: the point's longtitude (similarly scaled)
* Timestamp: The UNIX time (seconds) at which the point was added
* Entity Type: A categorial attribute, either a node (1), relation (2), or way (3)
* Primary Type: What type of landmark the point refers to, if any. A categorical attribute with 27
  options, including None (1).

## Query workload

The query workload is randomly generated to mimic analytics questions one might reasonably ask about
the data, such as:
* How many Buildings (Primary Type = 3) exist within a given lat-lon rectangle?
* How many nodes were added in a particular window of time?

In total, there are 7 types of queries, all of which compute a COUNT(*) aggregate.
The frequency of each type of query and its selectivity vary based on the random seed
`QUERY_GEN_SEED`.

## RL Agent


 
