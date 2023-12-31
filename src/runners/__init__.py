REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .pref_episode_runner import PrefEpisodeRunner
REGISTRY["pref_episode"] = PrefEpisodeRunner