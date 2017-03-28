\version "2.10.33"

\header {
  title = "142 - Schwing dich auf zu deinem Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  d4 a4 d4 e4 
  f4 d4 e2 
  c4 c4 c4 bes8 a8 
  g2 a2 
  d4 a4 d4 e4   %5
  f4 d4 e2 
  c4 c4 c4 bes8 a8 
  g2 a2 
  f'4 f4 e4 e4 
  d4 e4 cis2   %10
  a4 bes4 cis4 d4 
  d4 cis4 d2 
  f4 f4 e4 e4 
  d4 e4 cis2 
  a4 bes4 cis4 d4   %15
  d4 cis4 d2 
  
}

alto = \relative c'' {
  a4 a4 bes4 bes4 
  a4 g4 g2 
  a4 a8 g8 f4 f8 e8 
  d8 des8 d4 cis2 
  a'4 a4 g8 f8 e4   %5
  a8 g8 f4 e2 
  e4 a4 a4 g8 a8 
  bes4 c4 fis,2 
  a4 g4 g4 a4 
  a4 e4 a2   %10
  e4 a4 g4 a4 
  a4. g8 f2 
  a4 g4 g4 g4 
  f4 g4 e2 
  f4 f4 e4 d8 e8   %15
  f8 e16 d16 e4 fis2 
  
}

tenor = \relative c' {
  f4 e4 f4 g4 
  c,4 d4 c2 
  e4 f4 a,8 g8 f4 
  f4 e8 d8 e2 
  a4 d8 c8 bes4 b4   %5
  cis4 d8 c8 b2 
  c8 d8 ees4 d4 d4 
  d8 bes8 a8 g8 d'2 
  d4 d4 c4 c4 
  c8 b16 a16 b4 e2   %10
  cis4 d4 e4 f4 
  e2 d2 
  c4 d4 c8 bes8 a4 
  a4 bes4 a2 
  d4 d4 g,4 a8 bes8   %15
  a2 a2 
  
}

baixo = \relative c' {
  d4 c4 bes4 g4 
  a4 b4 c2 
  a8 g8 f8 e8 d4. c8 
  bes2 a2 
  f'4 fis4 g4 gis4   %5
  a4 bes8 a8 gis2 
  a4 g4 fis4 g8 f8 
  ees2 d2 
  d'8 c8 b8 g8 c8 bes8 a8 g8 
  fis4 gis4 a2   %10
  g4 f4 e4 d4 
  a'4 a,4 bes2 
  a4 b4 c4 cis4 
  d4 g,4 g'2 
  f4 e8 d8 e4 f8 g8   %15
  a4 a,4 d2 
  
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