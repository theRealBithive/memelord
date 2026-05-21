from django.core.management.base import BaseCommand

from ratings.scraper import populate_knn_tag_suggestions


class Command(BaseCommand):
    help = "Fill Image.knn_tag_suggestions by inheriting tags from visually-similar tagged images."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Max images to process (default: all without knn suggestions).",
        )
        parser.add_argument(
            "--refill",
            action="store_true",
            help="Recompute even for images that already have knn suggestions.",
        )
        parser.add_argument("--k", type=int, default=15, help="Neighbours to consider.")
        parser.add_argument(
            "--max-suggestions",
            type=int,
            default=8,
            help="Top-N tags to keep per image.",
        )
        parser.add_argument(
            "--min-similarity",
            type=float,
            default=0.5,
            help="Cosine similarity below which a neighbour is ignored.",
        )

    def handle(self, *args, **options):
        result = populate_knn_tag_suggestions(
            refill=options["refill"],
            limit=options["limit"],
            k=options["k"],
            max_suggestions=options["max_suggestions"],
            min_similarity=options["min_similarity"],
        )
        self.stdout.write(self.style.SUCCESS(f"Done. {result['updated']} updated."))
