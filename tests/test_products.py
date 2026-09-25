from datetime import UTC, datetime

from nwp_archivist.products import (
    ICON_ART_EU,
    ICON_D2,
    ICON_D2_EPS,
    ICON_EU_EPS,
    Product,
    expected_files,
    run_init_times,
    static_urls,
)

INIT = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _count(product: Product, variable: str) -> int:
    return sum(1 for f in expected_files(product, INIT) if f.field.variable == variable)


def test_icon_d2_eps_surface_field_has_20_members_of_49_steps() -> None:
    assert _count(ICON_D2_EPS, "T_2M") == 20 * 49


def test_icon_d2_eps_run_has_ten_variables() -> None:
    assert len(expected_files(ICON_D2_EPS, INIT)) == 10 * 20 * 49


def test_deterministic_urls_have_no_member_segment() -> None:
    urls = [f.url for f in expected_files(ICON_D2, INIT)]
    assert not any("/e/" in url for url in urls)


def test_ensemble_urls_have_a_member_segment() -> None:
    urls = [f.url for f in expected_files(ICON_D2_EPS, INIT)]
    assert all("/e/" in url for url in urls)


def test_icon_d2_radiation_has_193_steps_and_temperature_49() -> None:
    assert _count(ICON_D2, "ASWDIR_S") == 193
    assert _count(ICON_D2, "ASWDIFD_S") == 193
    assert _count(ICON_D2, "T_2M") == 49


def test_icon_art_eu_step_lists() -> None:
    assert _count(ICON_ART_EU, "TAOD_DUST") == 89
    assert _count(ICON_ART_EU, "ASOB_S_CS") == 64
    assert _count(ICON_ART_EU, "T_2M") == 75


def test_icon_eu_eps_step_lists() -> None:
    assert _count(ICON_EU_EPS, "T_2M") == 40 * 91
    assert _count(ICON_EU_EPS, "U_L72") == 40 * 64


def test_padded_step_axis_is_the_longest_field() -> None:
    assert ICON_D2.max_steps == 193
    assert ICON_EU_EPS.max_steps == 91


def test_surface_url_matches_the_provider_layout() -> None:
    first = expected_files(ICON_D2_EPS, INIT)[0]
    assert first.url == (
        "https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-eps/p/ASWDIR_S/r/"
        "2026-09-25T12%3A00/e/01/s/PT000H00M.grib2"
    )


def test_model_level_url_has_a_level_directory() -> None:
    urls = {f.url for f in expected_files(ICON_D2_EPS, INIT)}
    assert (
        "https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-eps/p/U/lvt1/150/lv1/63/r/"
        "2026-09-25T12%3A00/e/20/s/PT048H00M.grib2"
    ) in urls


def test_quarter_hour_step_file_name() -> None:
    urls = {f.url for f in expected_files(ICON_D2, INIT)}
    assert (
        "https://opendata.dwd.de/weather/nwp/v1/m/icon-d2/p/ASWDIR_S/r/"
        "2026-09-25T12%3A00/s/PT024H15M.grib2"
    ) in urls


def test_static_urls_of_an_ensemble_use_member_one() -> None:
    urls = static_urls(ICON_D2_EPS, INIT)
    assert urls["clat"].endswith("/p/CLAT/r/2026-09-25T12%3A00/e/01/s/PT000H00M.grib2")
    assert set(urls) == {"clat", "clon", "hsurf", "fr_land", "hhl_L62", "hhl_L63", "hhl_L64"}


def test_static_urls_of_a_deterministic_product_have_no_member() -> None:
    urls = static_urls(ICON_ART_EU, INIT)
    assert "/e/" not in urls["clat"]
    assert set(urls) == {"clat", "clon", "hsurf", "fr_land"}


def test_run_init_times_follow_the_cycle() -> None:
    times = run_init_times(
        ICON_EU_EPS,
        first=datetime(2026, 9, 24, 13, tzinfo=UTC),
        last=datetime(2026, 9, 25, 12, tzinfo=UTC),
    )
    assert [t.hour for t in times] == [18, 0, 6, 12]
    assert times[0].day == 24
