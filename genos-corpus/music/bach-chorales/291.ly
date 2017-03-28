\version "2.10.33"

\header {
  title = "291 - Was frag'ich nach der Welt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \partial 4 a4 
    fis4 e8 d8 a'4 a4 
  b2. b4 
  e,4 e4 a4 g4 
  fis4 e4 d4 a'4 
  b4 b4 a4 g4   %5
  fis2. e4 
  fis4 gis4 a4 b8 cis16 d16 
  cis4 b8 a8 a4 a4 
  a4 a4 d4 c4 
  b2. b4   %10
  b4 b4 e4 d4 
  cis2. a4 
  b4 a4 b4 cis4 
  d2. a4 
  a4 g8 fis8 e8 fis16 g16 e4   %15
  d2. 
}

alto = \relative c' {
  \partial 4 e4 
    d4 cis4 d4 fis4 
  g2. e4 
  e8 d8 cis4 d4 e4 
  e8 d4 cis8 a4 d4 
  d4 g4 g8 fis8 e4   %5
  d2. e4 
  d4 d4 cis4 fis4 
  e2 e4 e4 
  fis8 e8 fis8 g8 a4 fis4 
  g2. a4   %10
  a4 gis4 gis8 fis8 fis8 e8 
  e2. d4 
  d4 d4 g8 fis8 g4 
  fis2. e4 
  fis8 e8 d4 d4 cis4   %15
  a2. 
}

tenor = \relative c' {
  \partial 4 a4 
    a4 g4 a4 d4 
  d2. d4 
  cis8 b8 a4 a4 a4 
  a2 fis4 a4 
  g4 b4 cis8 d8 d8 cis8   %5
  a2. a4 
  a4 b4 a4 a4 
  a4 gis4 cis4 cis4 
  d4 d4 a4 d4 
  d2. fis4   %10
  e4 b4 b8 a8 a8 gis8 
  a2. a4 
  g4 a4 e'8 d8 e4 
  d2. e4 
  a,4 b4 b4 a8 g8   %15
  fis2. 
}

baixo = \relative c {
  \partial 4 cis4 
    d4 e4 fis4 d4 
  g2. gis4 
  a4 a8 g8 fis8 e8 d8 cis8 
  d4 a4 d4 fis4 
  g4 e4 a4 a,4   %5
  d2. cis4 
  d4 cis8 b8 fis'4 e8 d8 
  e2 a,4 a'4 
  d4 d,8 e8 fis4 d4 
  g2. dis4   %10
  e8 fis8 e8 d8 cis8 d8 e4 
  a,2. fis'4 
  g4 fis4 e4 ais,4 
  b2. cis4 
  d8 cis8 b8 a8 g4 a4   %15
  d2. 
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}