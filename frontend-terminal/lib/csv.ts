import type { CsvRecord } from "./api-types"

export function parseCsv(text: string): CsvRecord[] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"'
        i += 1
      } else if (ch === '"') quoted = false
      else field += ch
    } else if (ch === '"') quoted = true
    else if (ch === ",") {
      row.push(field)
      field = ""
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""))
      rows.push(row)
      row = []
      field = ""
    } else field += ch
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""))
    rows.push(row)
  }
  if (!rows.length) return []
  const headers = rows.shift() as string[]
  return rows
    .filter((item) => item.length > 1)
    .map((item) => {
      const record: CsvRecord = {}
      headers.forEach((header, index) => {
        record[header] = item[index] ?? ""
      })
      return record
    })
}
