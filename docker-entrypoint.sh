#!/bin/sh
# When JANULON_DATA is set, inject default paths for config, weights, db, output, and data_dir
# so a single mounted folder is used for all persistent state.
if [ -n "$JANULON_DATA" ]; then
  case "$1" in
    run)
      shift
      set -- run \
        --config "$JANULON_DATA/config.toml" \
        --weights "$JANULON_DATA/Janulon_weights.pkl" \
        --db "$JANULON_DATA/janulon.db" \
        --output_folder "$JANULON_DATA/output" \
        --data_dir "$JANULON_DATA" \
        "$@"
      ;;
    post)
      shift
      set -- post \
        --config "$JANULON_DATA/config.toml" \
        --db "$JANULON_DATA/janulon.db" \
        "$@"
      ;;
    import-data)
      shift
      set -- import-data \
        --data_dir "$JANULON_DATA" \
        --db "$JANULON_DATA/janulon.db" \
        "$@"
      ;;
    cleanup)
      shift
      set -- cleanup --db "$JANULON_DATA/janulon.db" "$@"
      ;;
    schedule)
      shift
      set -- schedule \
        --config "$JANULON_DATA/config.toml" \
        --weights "$JANULON_DATA/Janulon_weights.pkl" \
        --db "$JANULON_DATA/janulon.db" \
        --output_folder "$JANULON_DATA/output" \
        --data_dir "$JANULON_DATA" \
        "$@"
      ;;
  esac
fi
exec python -m main "$@"
