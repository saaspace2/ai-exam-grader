"""project_config.py - parse project_config_grader.yml into a typed object.

Marvel-style: this maps the YAML text file into a robust Python object, resolving
the environment (dev/acc/prd) so the same pipeline code can target any
environment just by changing the `env` argument.

This is SEPARATE from config.py (which holds the app's runtime settings like the
API key). project_config.py is specifically for the Databricks pipeline.
"""

from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class ProjectConfig:
    """Structured settings for the Databricks pipeline, per environment."""

    # Environment settings (resolved from dev/acc/prd)
    catalog_name: str
    schema_name: str

    # The Delta tables and Volume this pipeline manages
    tables: List[str]
    volume_name: str

    # Grading settings
    grader: str
    model_name: str

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Parse project_config_grader.yml and extract settings for `env`.

        Args:
            config_path: path to project_config_grader.yml
            env: one of "dev", "acc", "prd"
        """
        with open(config_path, "r") as file:
            raw = yaml.safe_load(file)

        # 1. Resolve environment-specific variables
        if env not in ["dev", "acc", "prd"]:
            raise ValueError(
                f"Unknown environment: {env}. Choose 'dev', 'acc', or 'prd'."
            )
        env_settings = raw[env]

        # 2. Map the file structure to the ProjectConfig fields
        return cls(
            catalog_name=env_settings["catalog_name"],
            schema_name=env_settings["schema_name"],
            tables=raw["tables"],
            volume_name=raw["volume_name"],
            grader=raw["grading"]["grader"],
            model_name=raw["grading"]["model_name"],
        )

    @property
    def base_path(self) -> str:
        """The catalog.schema prefix, e.g. 'exam_grader_dev.grading'."""
        return f"{self.catalog_name}.{self.schema_name}"