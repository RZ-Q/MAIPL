REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .globalRM_episode_runner import GlobalRMEpisodeRunner
REGISTRY["global_episode"] = GlobalRMEpisodeRunner

from .pref_episode_runner import PrefEpisodeRunner
REGISTRY["pref_episode"] = PrefEpisodeRunner