\version "2.10.33"

\header {
  title = "147 - Wenn ich in Angst und Not mein' Augen heb empor"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c' {
  \partial 4 ees4 
    bes'4 bes4 c4 d4 
  ees2 r4 d4 
  ees4 d4 c4 c4 
  bes2 r4 bes4 
  c4 bes4 aes4 g4   %5
  f2 r4 g4 
  ees4 f4 g8 aes8 bes4 
  aes2 g4 d'4 
  ees4 d4 c4 c4 
  bes2 r4 ees4   %10
  d4 c4 bes4 c4 
  bes4 aes4 g8 aes8 bes4 
  aes4 g4 f2 
  ees1 
  
}

alto = \relative c' {
  \partial 4 bes4 
    ees4 ees4 ees4 f4 
  g2 r4 f4 
  c'4 bes8 d,8 g4 f8 ees8 
  d2 r4 ees4 
  ees4 d4 ees8 bes8 bes4   %5
  bes2 r4 d4~ 
  d8 c4 bes8 bes4 ees4 
  ees8 f16 ees16 d4 ees4 f4 
  g8 f8 f4 g4 f8 ees8 
  d2 r4 g4   %10
  g4 ees4 ees4 ees4 
  f8 g4 f4 ees8 ees4 
  ees2 ees4 d4 
  bes1 
  
}

tenor = \relative c' {
  \partial 4 g8  aes8 
    bes8 c8 des4 c8 bes8 aes4 
  bes2 r4 bes4~ 
  bes8 a8 bes2 a4 
  f2 r4 bes4~ 
  bes8 aes4 g4 f4 ees8   %5
  d2 r4 g4 
  g4 f4 ees4 bes'4 
  c4 bes4 bes4 d4 
  c4 bes4 bes4 a4 
  f2 r4 c'4~   %10
  c8 bes4 aes4 g8 g4 
  d'8 c8 c4 c4 bes4 
  ees,8 f8 g8 ees8 c'4 bes8 aes8 
  g1 
  
}

baixo = \relative c {
  \partial 4 ees8  f8 
    g4 g4 aes8 g8 f4 
  ees2 r4 bes4 
  c4 g'4 ees4 f4 
  bes,2 r4 g4 
  aes4 bes4 c8 d8 ees4   %5
  bes2 r4 b4 
  c4 d4 ees4 g,4 
  f4 bes4 ees,4 bes''4~ 
  bes8 a8 bes4 ees,4 f4 
  bes,2 r4 c4   %10
  g'4 aes4 ees4 c4 
  d8 e8 f4 c4 g4 
  c4 bes4 aes4 bes4 
  ees,1 
  
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