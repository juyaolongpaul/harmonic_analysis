\version "2.10.33"

\header {
  title = "255 - Was frag' ich nach der Welt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \partial 4 a4 
    fis4 d4 a'4 a4 
  b2. b4 
  e,4 e4 a4 g4 
  fis4 d4 r4 a'4 
  b4 b4 a4 g4   %5
  fis2. e4 
  fis4 gis4 a4. b8 
  gis2 a4 a4 
  a4 a4 d4 c4 
  b2. b4   %10
  b4 b4 e4 d4 
  cis2. a4 
  b4 a4 b4 cis4 
  d2. a4 
  a4 g8 fis8 e4 e4   %15
  d2. 
}

alto = \relative c' {
  \partial 4 e4 
    d4 fis8 e8 d4 a'8 fis8 
  g2. d4 
  cis8 d8 cis8 b8 a4 e'4 
  d4 a4 r4 d4 
  d4 g4 fis4 e4   %5
  a,2. cis4 
  d4 e4 e4. fis8 
  e2 e4 e4 
  fis8 e8 fis8 g8 a8 g8 a8 fis8 
  g2. fis4   %10
  e4 e4 e4 e4 
  e2. d4 
  g4 a4 g4 b8 ais8 
  b2. a8 g8 
  fis4 e8 d8 d4 cis4   %15
  a2. 
}

tenor = \relative c' {
  \partial 4 a4 
    a4 a4 a4 d4 
  d2. g,4 
  g4. fis8 e4 a4 
  a4 fis4 r4 a4 
  g8 a8 b4 cis8 d8 d8 cis8   %5
  d2. cis4 
  a4 b4 a4. d8 
  b2 cis4 cis4 
  d4 d4 d4 d4 
  d2. b8 a8   %10
  gis8 fis8 gis8 a8 b8 a8 b8 gis8 
  a2. d4 
  d4 d4 d4 e4 
  fis2. e4 
  d4 a4 a8 b8 a8 g8   %15
  fis2. 
}

baixo = \relative c {
  \partial 4 cis4 
    d8 cis8 d8 e8 fis8 e8 fis8 d8 
  g2. g,4 
  a8 g8 a8 b8 cis8 b8 cis8 a8 
  d4 d4 r4 fis4 
  g4 e4 a4 a,4   %5
  d2. a4 
  d8 cis8 d8 b8 cis8 b8 cis8 d8 
  e4 e,4 a4 a4 
  d8 cis8 d8 e8 fis8 e8 fis8 d8 
  g2. dis4   %10
  e8 dis8 e8 fis8 gis8 fis8 gis8 e8 
  a2. fis4 
  g8 a8 g8 fis8 g8 fis8 g8 e8 
  b'2. cis,4 
  d8 e8 fis8 g8 a4 a,4   %15
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