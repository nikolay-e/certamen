import { test, expect } from "@playwright/test";

test.describe("GUI Application", () => {
  test("should load application successfully", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/Certamen GUI/);
  });

  test("should display sidebar with node categories", async ({ page }) => {
    await page.goto("/");
    const sidebar = page.locator(".sidebar");
    await expect(sidebar).toBeVisible();
  });

  test("should display canvas area", async ({ page }) => {
    await page.goto("/");
    const canvas = page.locator(".react-flow");
    await expect(canvas).toBeVisible();
  });

  test("should display properties panel", async ({ page }) => {
    await page.goto("/");
    const propertiesPanel = page.locator(".properties-panel");
    await expect(propertiesPanel).toBeVisible();
  });

  test("should display execute and cancel buttons in header", async ({
    page,
  }) => {
    await page.goto("/");
    const executeButton = page.getByRole("button", { name: /execute/i });
    const cancelButton = page.getByRole("button", { name: /cancel/i });
    await expect(executeButton).toBeVisible();
    await expect(cancelButton).toBeVisible();
  });
});
