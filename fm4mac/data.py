import cloudpickle
import grain
import lz4.frame

from fm4mac.graphs import make_batch


def save_dataset(dataset, path):
    with lz4.frame.open(path, 'wb') as f:
        f.write(cloudpickle.dumps(dataset))


def load_dataset(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.loads(f.read())


def prebatch(graphs, max_edges, n_batches, rng):
    """
    Shuffled batches of at most `max_edges` edges (attention is quadratic in the number of edges of a batch).
    """

    batches, current, total = [], [], 0

    while len(batches) < n_batches:
        for i in rng.permutation(len(graphs)):
            n = int(graphs[i].n_edge.sum())

            if current and total + n > max_edges:
                batches.append(make_batch([graphs[j] for j in current]))
                current, total = [], 0

                if len(batches) == n_batches:
                    break

            current.append(i)
            total += n

    return batches


def create_data_loader(items, batch_size, seed, map_fn=lambda x: x, batch_fn=make_batch, num_workers=0):
    """
    Infinite shuffled loader of batches of `map_fn(item)`; set the number of steps explicitly.
    """

    ds = grain.MapDataset.source(items).seed(seed).shuffle().map(map_fn)
    ds = ds.batch(batch_size, batch_fn=batch_fn).repeat()
    ds = ds.to_iter_dataset(grain.ReadOptions(num_threads=0))
    return ds.mp_prefetch(grain.MultiprocessingOptions(num_workers, per_worker_buffer_size=16))
