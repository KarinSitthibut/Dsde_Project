import puppeteer from "puppeteer";
import fs from "fs";
import path from "path";

async function getLatLon(page, shortUrl) {
    try {
        await page.goto(shortUrl, { waitUntil: "networkidle2" });

        const finalUrl = page.url();
        const match = finalUrl.match(/@([0-9.\-]+),([0-9.\-]+)/);

        if (!match) return { lat: null, lon: null };

        return {
            lat: match[1],
            lon: match[2],
        };
    } catch (e) {
        return { lat: null, lon: null };
    }
}

async function main() {
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();

    console.log("Loading main page…");
    await page.goto("https://pr-bangkok.com/?p=77768", { waitUntil: "networkidle2" });

    const html = await page.content();

    // Extract all blocks
    const items = await page.$$eval(".item", (nodes) =>
        nodes.map((el) => {
            const name = el.querySelector("strong")?.innerText?.trim() || "";
            const map = el.querySelector(".map-link")?.href || "";
            return { name, map };
        })
    );

    console.log(`Found ${items.length} districts`);

    // Prepare output array
    const results = [];

    for (let item of items) {
        console.log("Resolving map for:", item.name);

        const { lat, lon } = await getLatLon(page, item.map);

        results.push({
            district: item.name.replace("♦", "").trim(),
            map_url: item.map,
            lat,
            lon,
        });
    }

    // Save CSV
    const csv = [
        "district,map_url,lat,lon",
        ...results.map((r) => `${r.district},${r.map_url},${r.lat},${r.lon}`),
    ].join("\n");

    const outPath = path.resolve("../data/external/district_offices.csv");
    fs.writeFileSync(outPath, csv, "utf8");

    console.log("Saved →", outPath);
    await browser.close();
}

main();
